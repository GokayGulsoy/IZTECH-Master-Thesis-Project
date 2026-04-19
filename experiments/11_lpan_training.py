#!/usr/bin/env python3
"""
Experiment 11: LPAN — Learnable Polynomial Activation Networks
===============================================================
Ultimate contribution: makes polynomial coefficients learnable parameters
warm-started from weighted minimax, then fine-tuned with knowledge
distillation from the baseline teacher model.

Pipeline:
  1. Load baseline (teacher) model
  2. Profile activations → fit initial polynomial coefficients
  3. Create LPAN student model (coefficients as nn.Parameters)
  4. Distillation training: task_loss + KL(teacher||student) + L2(coeffs)
  5. Evaluate improvement over fixed-polynomial replacement

Outputs → results/lpan/
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

from fhe_thesis.config import MODEL_REGISTRY, RESULTS_DIR, ensure_dirs
from fhe_thesis.models.profiling import profile_model, compute_poly_coefficients
from fhe_thesis.models.replacement import replace_activations
from fhe_thesis.training.trainer import compute_metrics, detect_device

# Output directory
LPAN_DIR = RESULTS_DIR / "lpan"


class LPANDistillationTrainer(Trainer):
    """Custom Trainer that adds KL distillation loss from teacher model."""

    def __init__(self, teacher_model, alpha=1.0, beta=0.01, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # Task loss
        task_loss = F.cross_entropy(student_logits, labels)

        # KL distillation
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        # L2 on polynomial coefficients
        l2_reg = torch.tensor(0.0, device=student_logits.device)
        for name, param in model.named_parameters():
            if "coeffs" in name:
                l2_reg = l2_reg + param.pow(2).sum()

        loss = task_loss + self.alpha * kl_loss + self.beta * l2_reg

        inputs["labels"] = labels
        return (loss, student_outputs) if return_outputs else loss


def run_lpan_experiment(
    model_key: str,
    epochs: int = 5,
    degree: int = 8,
    profile_samples: int = 1000,
    alpha: float = 1.0,
    beta: float = 0.01,
    temperature: float = 2.0,
    learnable_interval: bool = False,
):
    """Run LPAN distillation for a single model."""
    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["name"]
    short = cfg["short"]
    num_layers = cfg["layers"]
    hidden = cfg["hidden"]
    bs = cfg["batch_size"]
    lr = cfg["lr"]

    print(f"\n{'='*70}")
    print(f"  LPAN Experiment: {short}")
    print(f"  alpha={alpha}, beta={beta}, T={temperature}")
    print(f"{'='*70}")

    model_dir = LPAN_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1/6] Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2")

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True,
                        padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_ds, eval_ds = tokenized["train"], tokenized["validation"]

    # 2. Train baseline (teacher)
    print("\n[2/6] Training baseline (teacher)...")
    teacher = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = detect_device()

    teacher_args = TrainingArguments(
        output_dir=str(model_dir / "teacher"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr,
        weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=100, report_to="none",
        fp16=torch.cuda.is_available(),
        no_cuda=device.type == "cpu",
    )
    teacher_trainer = Trainer(
        model=teacher, args=teacher_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    teacher_trainer.train()
    teacher_eval = teacher_trainer.evaluate()
    teacher_acc = teacher_eval["eval_accuracy"]
    print(f"  Teacher accuracy: {teacher_acc:.4f}")

    # 3. Profile activations
    print("\n[3/6] Profiling activations...")
    profile_data = profile_model(model_name, num_layers, profile_samples)

    # 4. Fit initial polynomial coefficients
    print("\n[4/6] Fitting initial polynomials (warm-start)...")
    poly_coeffs = compute_poly_coefficients(profile_data, num_layers, degree)

    # 5a. Fixed-polynomial baseline (for comparison)
    print("\n[5/6] Running fixed-polynomial baseline...")
    fixed_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    fixed_model.load_state_dict(teacher.state_dict())
    replace_activations(fixed_model, poly_coeffs, hidden)

    use_fp16_poly = num_layers <= 4
    fixed_args = TrainingArguments(
        output_dir=str(model_dir / "fixed_poly"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr * 0.5,
        weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=100, report_to="none",
        fp16=use_fp16_poly and torch.cuda.is_available(),
        no_cuda=device.type == "cpu",
        max_grad_norm=1.0,
    )
    fixed_trainer = Trainer(
        model=fixed_model, args=fixed_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    fixed_trainer.train()
    fixed_eval = fixed_trainer.evaluate()
    fixed_acc = fixed_eval["eval_accuracy"]
    print(f"  Fixed-poly accuracy: {fixed_acc:.4f}")

    # 5b. LPAN distillation (our contribution)
    print("\n[6/6] LPAN distillation training...")
    lpan_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    lpan_model.load_state_dict(teacher.state_dict())
    replace_activations(lpan_model, poly_coeffs, hidden, learnable=True)

    # Move teacher to same device
    teacher.to(device)

    lpan_args = TrainingArguments(
        output_dir=str(model_dir / "lpan"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr * 0.5,
        weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=100, report_to="none",
        fp16=use_fp16_poly and torch.cuda.is_available(),
        no_cuda=device.type == "cpu",
        max_grad_norm=1.0,
    )

    lpan_trainer = LPANDistillationTrainer(
        teacher_model=teacher,
        alpha=alpha, beta=beta, temperature=temperature,
        model=lpan_model, args=lpan_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    lpan_trainer.train()
    lpan_eval = lpan_trainer.evaluate()
    lpan_acc = lpan_eval["eval_accuracy"]
    print(f"  LPAN accuracy: {lpan_acc:.4f}")

    # Log learned coefficient changes
    coeff_changes = {}
    for name, param in lpan_model.named_parameters():
        if "coeffs" in name:
            coeff_changes[name] = {
                "values": param.data.cpu().tolist(),
                "norm": float(param.data.norm()),
            }

    result = {
        "model": model_name, "short": short,
        "layers": num_layers, "hidden": hidden,
        "teacher_acc": teacher_acc,
        "fixed_poly_acc": fixed_acc,
        "lpan_acc": lpan_acc,
        "fixed_drop": (teacher_acc - fixed_acc) * 100,
        "lpan_drop": (teacher_acc - lpan_acc) * 100,
        "improvement": (lpan_acc - fixed_acc) * 100,
        "alpha": alpha, "beta": beta, "temperature": temperature,
        "poly_degree": degree, "epochs": epochs,
        "learnable_interval": learnable_interval,
        "coefficient_changes": coeff_changes,
    }

    with open(model_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Cleanup
    del teacher, fixed_model, lpan_model
    torch.cuda.empty_cache()

    return result


def plot_lpan_comparison(all_results: List[Dict]):
    """Comparison chart: Teacher vs Fixed vs LPAN."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = [r["short"] for r in all_results]
    teacher = [r["teacher_acc"] * 100 for r in all_results]
    fixed = [r["fixed_poly_acc"] * 100 for r in all_results]
    lpan = [r["lpan_acc"] * 100 for r in all_results]

    x = np.arange(len(models))
    w = 0.25

    axes[0].bar(x - w, teacher, w, label="Teacher (Baseline)", color="tab:blue", alpha=0.85)
    axes[0].bar(x, fixed, w, label="Fixed Polynomial", color="tab:orange", alpha=0.85)
    axes[0].bar(x + w, lpan, w, label="LPAN (Ours)", color="tab:green", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("SST-2: Teacher vs Fixed-Poly vs LPAN")
    axes[0].legend(); axes[0].set_ylim(75, 95); axes[0].grid(True, alpha=0.3, axis="y")

    fixed_drops = [r["fixed_drop"] for r in all_results]
    lpan_drops = [r["lpan_drop"] for r in all_results]
    axes[1].bar(x - w/2, fixed_drops, w, label="Fixed Poly Drop", color="tab:orange", alpha=0.85)
    axes[1].bar(x + w/2, lpan_drops, w, label="LPAN Drop", color="tab:green", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Accuracy Drop (%)")
    axes[1].set_title("Accuracy Drop from Baseline")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(LPAN_DIR / "lpan_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved lpan_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="LPAN Distillation Experiments")
    parser.add_argument("--models", nargs="+", default=["tiny"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--profile-samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.0, help="KL distillation weight")
    parser.add_argument("--beta", type=float, default=0.01, help="L2 coefficient regularization")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--learnable-interval", action="store_true",
                        help="Also learn interval boundaries")
    args = parser.parse_args()

    ensure_dirs()
    LPAN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  LPAN: Learnable Polynomial Activation Networks")
    print(f"  Models: {args.models}, alpha={args.alpha}, beta={args.beta}, T={args.temperature}")
    print("=" * 70)

    all_results = []
    for model_key in args.models:
        result = run_lpan_experiment(
            model_key, args.epochs, args.degree, args.profile_samples,
            args.alpha, args.beta, args.temperature, args.learnable_interval,
        )
        all_results.append(result)

    # Save combined results
    with open(LPAN_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    if len(all_results) >= 1:
        plot_lpan_comparison(all_results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  LPAN RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Teacher':>10} {'Fixed':>10} {'LPAN':>10} "
          f"{'Fixed Drop':>12} {'LPAN Drop':>12} {'Improvement':>12}")
    print(f"  {'-'*81}")
    for r in all_results:
        print(f"  {r['short']:<15} {r['teacher_acc']:>9.4f} {r['fixed_poly_acc']:>9.4f} "
              f"{r['lpan_acc']:>9.4f} {r['fixed_drop']:>11.2f}% "
              f"{r['lpan_drop']:>11.2f}% {r['improvement']:>+11.2f}%")


if __name__ == "__main__":
    main()
