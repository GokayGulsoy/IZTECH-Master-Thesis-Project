#!/usr/bin/env python3
"""
Experiment 10: Multi-Dataset Evaluation (SST-2, MRPC, QNLI)
=============================================================
Extends polynomial replacement to three GLUE tasks, demonstrating
generality of the approach.

Outputs → results/multi_dataset/
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    EvalPrediction, Trainer, TrainingArguments,
)

from fhe_thesis.config import DEFAULT_MODEL_NAME, MULTI_DATASET_DIR, ensure_dirs
from fhe_thesis.models.replacement import replace_activations
from fhe_thesis.models.profiling import profile_model, compute_poly_coefficients
from fhe_thesis.training.trainer import detect_device

OUT = MULTI_DATASET_DIR

TASK_CONFIGS = {
    "sst2": {
        "glue_name": "sst2", "num_labels": 2,
        "text_fields": ["sentence"], "label_col": "label",
        "metric": "accuracy", "epochs": 5, "lr": 3e-5,
        "description": "Sentiment Analysis (SST-2)",
    },
    "mrpc": {
        "glue_name": "mrpc", "num_labels": 2,
        "text_fields": ["sentence1", "sentence2"], "label_col": "label",
        "metric": "f1", "epochs": 8, "lr": 3e-5,
        "description": "Paraphrase Detection (MRPC)",
    },
    "qnli": {
        "glue_name": "qnli", "num_labels": 2,
        "text_fields": ["question", "sentence"], "label_col": "label",
        "metric": "accuracy", "epochs": 5, "lr": 3e-5,
        "description": "Question NLI (QNLI)",
    },
}


def compute_accuracy(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": float(np.mean(preds == eval_pred.label_ids))}


def compute_f1_and_accuracy(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    accuracy = float(np.mean(preds == labels))
    tp = float(np.sum((preds == 1) & (labels == 1)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": accuracy, "f1": f1, "precision": prec, "recall": rec}


def load_task_data(task_name, tokenizer, max_length=128):
    cfg = TASK_CONFIGS[task_name]
    dataset = load_dataset("glue", cfg["glue_name"])
    fields = cfg["text_fields"]

    def tok_fn(examples):
        if len(fields) == 1:
            return tokenizer(examples[fields[0]], truncation=True,
                            padding="max_length", max_length=max_length)
        return tokenizer(examples[fields[0]], examples[fields[1]],
                        truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tok_fn, batched=True)
    tokenized = tokenized.rename_column(cfg["label_col"], "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized["train"], tokenized["validation"]


def train_and_evaluate(task_name, tokenizer, variant="baseline",
                       poly_coeffs=None, base_state_dict=None):
    cfg = TASK_CONFIGS[task_name]
    device = detect_device()
    metric_fn = compute_f1_and_accuracy if cfg["metric"] == "f1" else compute_accuracy
    train_data, eval_data = load_task_data(task_name, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME, num_labels=cfg["num_labels"],
    )
    if base_state_dict:
        model.load_state_dict(base_state_dict, strict=False)
    if poly_coeffs and variant in ("poly_zero_shot", "poly_finetuned"):
        replace_activations(model, poly_coeffs)

    output_dir = OUT / task_name / variant
    if variant == "poly_zero_shot":
        args = TrainingArguments(output_dir=str(output_dir),
                                per_device_eval_batch_size=64,
                                report_to="none", no_cuda=device.type == "cpu")
        trainer = Trainer(model=model, args=args, eval_dataset=eval_data,
                         compute_metrics=metric_fn)
        results = trainer.evaluate()
        return {
            "task": task_name, "variant": variant,
            "primary_metric": results.get(f"eval_{cfg['metric']}", 0),
            "metric_name": cfg["metric"],
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=cfg["lr"], warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model=cfg["metric"],
        save_total_limit=1, report_to="none",
        no_cuda=device.type == "cpu", fp16=device.type == "cuda",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data,
                     eval_dataset=eval_data, compute_metrics=metric_fn)
    trainer.train()
    results = trainer.evaluate()
    trainer.save_model(str(output_dir / "best_model"))

    return {
        "task": task_name, "variant": variant,
        "primary_metric": results.get(f"eval_{cfg['metric']}", 0),
        "metric_name": cfg["metric"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "qnli"],
                        choices=["sst2", "mrpc", "qnli"])
    args = parser.parse_args()

    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Multi-Dataset Evaluation: GLUE Benchmark")
    print(f"  Tasks: {args.tasks}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    # Profile and compute poly coefficients
    print("\n[1/3] Profiling + computing polynomial coefficients...")
    profile_data = profile_model(DEFAULT_MODEL_NAME, num_layers=2, num_samples=1000)
    poly_coeffs = compute_poly_coefficients(profile_data, num_layers=2, degree=8)

    all_results = {}
    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {TASK_CONFIGS[task_name]['description']}")
        print(f"{'='*60}")

        print("\n  [2/3] Training baseline...")
        baseline = train_and_evaluate(task_name, tokenizer, "baseline")
        print(f"    Baseline {baseline['metric_name']}: {baseline['primary_metric']:.4f}")

        print("  [2/3] Zero-shot polynomial evaluation...")
        zero_shot = train_and_evaluate(task_name, tokenizer, "poly_zero_shot",
                                       poly_coeffs=poly_coeffs)
        print(f"    Zero-shot: {zero_shot['primary_metric']:.4f}")

        print("  [2/3] Fine-tuning polynomial model...")
        finetuned = train_and_evaluate(task_name, tokenizer, "poly_finetuned",
                                        poly_coeffs=poly_coeffs)
        print(f"    Fine-tuned: {finetuned['primary_metric']:.4f}")

        all_results[task_name] = {
            "baseline": baseline, "poly_zero_shot": zero_shot, "poly_finetuned": finetuned,
        }

    # Plot
    print("\n[3/3] Generating plots...")
    tasks = sorted(all_results.keys())
    variants = ["baseline", "poly_zero_shot", "poly_finetuned"]
    labels = ["Baseline", "Poly (Zero-shot)", "Poly (Fine-tuned)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tasks))
    width = 0.25
    for i, (var, label, color) in enumerate(zip(variants, labels, colors)):
        vals = [all_results[t].get(var, {}).get("primary_metric", 0) for t in tasks]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels([TASK_CONFIGS[t]["description"] for t in tasks])
    ax.set(ylabel="Primary Metric", title="Multi-Task: Baseline vs Polynomial Replacement")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "multi_task_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved multi_task_comparison.png")

    with open(OUT / "multi_dataset_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  All results saved to: {OUT}/")


if __name__ == "__main__":
    main()
