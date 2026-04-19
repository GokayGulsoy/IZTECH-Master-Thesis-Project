#!/usr/bin/env python3
"""
Experiment 05: Multi-Model Scaling Evaluation
===============================================
Runs the full pipeline (profile → fit → replace → train → eval) on
BERT-Tiny/Mini/Small/Base for reproducible scaling analysis.

Outputs → results/multi_model/
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR, ensure_dirs
from fhe_thesis.tasks import GLUE_TASKS, get_task
from fhe_thesis.models.profiling import profile_model, compute_poly_coefficients
from fhe_thesis.models.replacement import replace_activations
from fhe_thesis.training.trainer import (
    compute_metrics_for_task,
    load_glue_dataset,
    train_and_eval,
    calibrate_grad_norm,
    distill_and_eval,
)


def _task_subpath(task_name: str) -> str:
    """SST-2 keeps the legacy layout (no subdir); other tasks add a subdir.

    This preserves any baselines that were already trained for SST-2 under
    `results/multi_model/<model>/baseline/...` while giving each new GLUE
    task its own isolated directory `results/multi_model/<model>/<task>/...`.
    """
    return "" if task_name == "sst2" else task_name


def run_model(
    model_key: str,
    cfg: dict,
    epochs: int,
    degree: int,
    profile_samples: int,
    task_name: str = "sst2",
):
    """Run the full pipeline for a single model variant."""
    model_name = cfg["name"]
    short = cfg["short"]
    num_layers = cfg["layers"]
    hidden = cfg["hidden"]
    bs = cfg["batch_size"]
    lr = cfg["lr"]

    task = get_task(task_name)
    metric_fn = compute_metrics_for_task(task)
    metric_for_best = task.metric_for_best_model
    primary_key = task.metric

    print(f"\n{'='*70}")
    print(f"  {short} ({model_name}) \u2014 task: {task.description}")
    print(f"  {num_layers} layers, {hidden} hidden, ~{cfg['params_m']}M params")
    print(f"{'='*70}")

    model_dir = MULTI_MODEL_DIR / model_key / _task_subpath(task_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, eval_dict = load_glue_dataset(task, tokenizer, max_length=128)
    # Best-model selection runs on the first eval split (matched for MNLI).
    primary_eval_name = next(iter(eval_dict))
    eval_ds = eval_dict[primary_eval_name]

    # Baseline
    print("\n[2/5] Training BASELINE...")
    baseline_kwargs = {"num_labels": task.num_labels}
    if task.is_regression:
        baseline_kwargs["problem_type"] = "regression"
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **baseline_kwargs
    )
    baseline_res = train_and_eval(
        baseline_model,
        train_ds,
        eval_ds,
        str(model_dir / "baseline"),
        epochs,
        bs,
        lr,
        f"{short} Baseline",
        compute_metrics_fn=metric_fn,
        metric_for_best_model=metric_for_best,
    )

    # Profile — use the fine-tuned baseline model so polynomial intervals
    # match the actual activation distributions (critical for Base depth).
    print("\n[3/5] Profiling activations...")
    best_path = model_dir / "baseline" / "best_model"
    if best_path.exists():
        profile_obj = AutoModelForSequenceClassification.from_pretrained(
            str(best_path), **baseline_kwargs
        )
    else:
        profile_obj = baseline_model
    profile_data = profile_model(
        model_name, num_layers, profile_samples, model_obj=profile_obj
    )
    # Free profiling copy if we loaded one
    if best_path.exists():
        del profile_obj
        torch.cuda.empty_cache()

    # Fit polynomials
    print("\n[4/5] Fitting weighted minimax polynomials...")
    poly_coeffs = compute_poly_coefficients(profile_data, num_layers, degree)

    # Poly-replaced + fine-tune
    print("\n[5/5] Polynomial replacement + fine-tuning...")
    if best_path.exists():
        poly_model = AutoModelForSequenceClassification.from_pretrained(
            str(best_path), **baseline_kwargs
        )
    else:
        poly_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **baseline_kwargs
        )
        poly_model.load_state_dict(baseline_model.state_dict())

    replace_activations(poly_model, poly_coeffs, hidden)

    # Zero-shot eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poly_model.to(device)
    zs_args = TrainingArguments(
        output_dir=str(model_dir / "poly_zs"),
        per_device_eval_batch_size=64,
        report_to="none",
        no_cuda=device.type == "cpu",
    )
    zs_trainer = Trainer(
        model=poly_model, args=zs_args, eval_dataset=eval_ds, compute_metrics=metric_fn
    )
    zs_eval = zs_trainer.evaluate()
    zs_acc = zs_eval[f"eval_{primary_key}"]
    print(f"  Zero-shot {primary_key}: {zs_acc:.4f}")

    # Fine-tune with knowledge distillation from the standard-activation teacher.
    # Polynomial activations alter the loss landscape; KD provides rich
    # soft-target gradients that guide the student past trivial local minima.
    use_fp16_poly = num_layers <= 4
    lr_scale = 0.5

    # Calibrate gradient clipping for polynomial activations.
    calibrated_norm = calibrate_grad_norm(
        poly_model,
        train_ds,
        batch_size=bs,
        num_batches=20,
        percentile=90.0,
    )

    # Load the baseline teacher model for distillation (on CPU to save GPU RAM).
    if best_path.exists():
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            str(best_path), **baseline_kwargs
        )
    else:
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **baseline_kwargs
        )
        teacher_model.load_state_dict(baseline_model.state_dict())
    teacher_model.eval()

    ft_res = distill_and_eval(
        student_model=poly_model,
        teacher_model=teacher_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=str(model_dir / "poly_ft"),
        epochs=epochs,
        batch_size=bs,
        lr=lr * lr_scale,
        label=f"{short} Poly Fine-Tuned",
        use_fp16=use_fp16_poly,
        max_grad_norm=calibrated_norm,
        alpha=0.5,
        temperature=4.0,
        compute_metrics_fn=metric_fn,
        metric_for_best_model=metric_for_best,
    )
    del teacher_model
    torch.cuda.empty_cache()

    # Compute depth
    total_depth = 0
    for li in range(num_layers):
        for op in ["GELU", "Softmax", "LN"]:
            k = f"L{li}_{op}"
            if k in poly_coeffs:
                d = poly_coeffs[k]["degree"]
                total_depth += max(1, math.ceil(math.log2(d + 1)))

    result = {
        "model": model_name,
        "short": short,
        "task": task_name,
        "primary_metric": primary_key,
        "layers": num_layers,
        "hidden": hidden,
        "params_m": cfg["params_m"],
        "baseline_acc": baseline_res["accuracy"],
        "zero_shot_acc": zs_acc,
        "finetuned_acc": ft_res["accuracy"],
        "accuracy_drop": baseline_res["accuracy"] - ft_res["accuracy"],
        "accuracy_drop_pct": (baseline_res["accuracy"] - ft_res["accuracy"]) * 100,
        "poly_degree": degree,
        "total_depth": total_depth,
        "epochs": epochs,
    }

    with open(model_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    del baseline_model, poly_model
    torch.cuda.empty_cache()
    return result


def plot_scaling_analysis(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = [r["short"] for r in all_results]
    params = [r["params_m"] for r in all_results]
    baseline = [r["baseline_acc"] * 100 for r in all_results]
    finetuned = [r["finetuned_acc"] * 100 for r in all_results]
    drops = [r["accuracy_drop_pct"] for r in all_results]
    layers = [r["layers"] for r in all_results]

    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, baseline, w, label="Baseline", color="tab:blue", alpha=0.85)
    axes[0].bar(
        x + w / 2, finetuned, w, label="Poly Fine-Tuned", color="tab:green", alpha=0.85
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("SST-2 Accuracy")
    axes[0].legend()
    axes[0].set_ylim(75, 95)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, (b, f) in enumerate(zip(baseline, finetuned)):
        axes[0].text(i - w / 2, b + 0.3, f"{b:.1f}%", ha="center", fontsize=8)
        axes[0].text(i + w / 2, f + 0.3, f"{f:.1f}%", ha="center", fontsize=8)

    axes[1].plot(layers, drops, "o-", color="tab:red", linewidth=2, markersize=8)
    for l, d, m in zip(layers, drops, models):
        axes[1].annotate(
            m, (l, d), textcoords="offset points", xytext=(10, 5), fontsize=8
        )
    axes[1].set(xlabel="Layers", ylabel="Accuracy Drop (%)", title="Drop vs Depth")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(params, drops, "s-", color="tab:purple", linewidth=2, markersize=8)
    for p, d, m in zip(params, drops, models):
        axes[2].annotate(
            m, (p, d), textcoords="offset points", xytext=(10, 5), fontsize=8
        )
    axes[2].set(
        xlabel="Parameters (M)", ylabel="Accuracy Drop (%)", title="Drop vs Size"
    )
    axes[2].set_xscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MULTI_MODEL_DIR / "scaling_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved scaling_analysis.png")


def plot_literature_comparison(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r["short"] for r in all_results]
    drops = [r["accuracy_drop_pct"] for r in all_results]
    lit_models = ["Iron\n(BERT-Base)", "THE-X\n(BERT-Base)", "BOLT\n(BERT-Base)"]
    lit_drops = [2.0, 3.5, 0.8]
    all_names = models + lit_models
    all_drops = drops + lit_drops
    colors = ["tab:green"] * len(models) + ["tab:gray"] * len(lit_models)
    bars = ax.bar(
        all_names, all_drops, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )
    for bar, d in zip(bars, all_drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{d:.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title("SST-2 Accuracy Drop: Our Method vs Literature")
    ax.grid(True, alpha=0.2, axis="y")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="tab:green", label="Ours (Weighted Minimax + Adaptive)"),
            Patch(facecolor="tab:gray", label="Literature"),
        ],
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(
        MULTI_MODEL_DIR / "literature_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  Saved literature_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Multi-model scaling evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny", "mini", "small", "base"],
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--task",
        default="sst2",
        choices=list(GLUE_TASKS),
        help="GLUE task to train baselines + poly-FT for (default: sst2)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--profile-samples", type=int, default=1000)
    args = parser.parse_args()

    ensure_dirs()
    print("=" * 70)
    print(
        f"  Multi-Model Scaling Evaluation — task: {args.task}, models: {args.models}"
    )
    print("=" * 70)

    all_results = []
    for model_key in args.models:
        cfg = MODEL_REGISTRY[model_key]
        result = run_model(
            model_key,
            cfg,
            args.epochs,
            args.degree,
            args.profile_samples,
            task_name=args.task,
        )
        all_results.append(result)

    suffix = "" if args.task == "sst2" else f"_{args.task}"
    out_file = MULTI_MODEL_DIR / f"all_results{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    if len(all_results) >= 2:
        plot_scaling_analysis(all_results)
        plot_literature_comparison(all_results)

    print(f"\n{'='*70}")
    print(
        f"  {'Model':<15} {'Layers':>6} {'Params':>8} {'Baseline':>10} "
        f"{'Poly-FT':>10} {'Drop':>8} {'Depth':>6}"
    )
    print(f"  {'-'*63}")
    for r in all_results:
        print(
            f"  {r['short']:<15} {r['layers']:>6} {r['params_m']:>7.1f}M "
            f"{r['baseline_acc']:>9.4f} {r['finetuned_acc']:>9.4f} "
            f"{r['accuracy_drop_pct']:>7.2f}% {r['total_depth']:>5}"
        )
    print(f"\n  Results saved to {MULTI_MODEL_DIR}/")


if __name__ == "__main__":
    main()
