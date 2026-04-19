#!/usr/bin/env python3
"""
Generate all thesis figures from saved experiment results.
===========================================================
Recreates every figure used in the thesis from the results/ directory.
Run individual experiments first, then this script to regenerate figures.

Usage::
    python experiments/generate_figures.py           # all figures
    python experiments/generate_figures.py --only 1  # specific figure set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fhe_thesis.config import (
    RESULTS_DIR,
    POLY_APPROX_DIR,
    ACTIVATION_PROFILES_DIR,
    BSGS_EVAL_DIR,
    ENCRYPTED_INFERENCE_DIR,
    ERROR_PROPAGATION_DIR,
    MODEL_REGISTRY,
    MULTI_MODEL_DIR,
    MULTI_DATASET_DIR,
    ensure_dirs,
)

FIG_DIR = RESULTS_DIR / "figures"


def fig_poly_approx():
    """Figure set 1: Polynomial approximation comparisons."""
    json_path = POLY_APPROX_DIR / "numerical_results.json"
    if not json_path.exists():
        print(
            "  [SKIP] No poly_approx results found. Run: python experiments/run_analysis.py poly"
        )
        return
    with open(json_path) as f:
        json.load(f)
    pngs = sorted(POLY_APPROX_DIR.glob("*_approximations_deg*.png"))
    if pngs:
        for p in pngs:
            print(f"  [OK] {p}")
    else:
        print(
            "  [OK] numerical_results.json present (per-degree PNGs not yet rendered)"
        )


def fig_activation_profiles():
    """Figure set 2: Activation distribution histograms — per model."""
    any_found = False
    for key, cfg in MODEL_REGISTRY.items():
        png = ACTIVATION_PROFILES_DIR / key / "activation_distributions.png"
        if png.exists():
            any_found = True
            print(f"  [OK] {cfg['short']:<10s} → {png}")
    if not any_found:
        print("  [SKIP] Run: python experiments/run_analysis.py profile")


def fig_bsgs():
    """Figure set 3: BSGS polynomial evaluation comparison."""
    png = BSGS_EVAL_DIR / "bsgs_comparison.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run: python experiments/run_analysis.py bsgs")


def fig_error_propagation():
    """Figure set 4: Error propagation analysis — per model."""
    any_found = False
    for key, cfg in MODEL_REGISTRY.items():
        png = ERROR_PROPAGATION_DIR / key / "error_propagation.png"
        if png.exists():
            any_found = True
            print(f"  [OK] {cfg['short']:<10s} → {png}")
    if not any_found:
        print("  [SKIP] Run: python experiments/run_analysis.py error")


def fig_multi_model():
    """Figure set 5: Multi-model accuracy across GLUE tasks."""
    json_path = MULTI_MODEL_DIR / "scaling_results.json"
    if not json_path.exists():
        print(
            "  [SKIP] Run: python run_staged_lpan.py --model <key> --task <task>"
            " (then aggregate into scaling_results.json)"
        )
        return

    with open(json_path) as f:
        data = json.load(f)

    # Collect accuracies by (model, task). The aggregator records may be a
    # flat list of {short, task, baseline_acc, poly_acc, params_m} entries.
    if isinstance(data, list):
        records = data
    else:
        records = data.get("records", [])

    tasks = sorted({r.get("task", "sst2") for r in records})
    models_order = [cfg["short"] for cfg in MODEL_REGISTRY.values()]
    n_tasks = len(tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5), squeeze=False)
    for ti, task in enumerate(tasks):
        ax = axes[0, ti]
        task_records = [r for r in records if r.get("task", "sst2") == task]
        present = [r["short"] for r in task_records]
        baseline = [r.get("baseline_acc", 0) * 100 for r in task_records]
        poly = [r.get("poly_acc", 0) * 100 for r in task_records]
        x = np.arange(len(present))
        w = 0.35
        ax.bar(x - w / 2, baseline, w, label="Baseline", color="tab:blue", alpha=0.85)
        ax.bar(x + w / 2, poly, w, label="LPAN", color="tab:green", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=15)
        ax.set(ylabel="Accuracy (%)", title=f"{task.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "LPAN vs Baseline across BERT variants and GLUE tasks", fontweight="bold"
    )
    plt.tight_layout()
    out = FIG_DIR / "multi_model_glue.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}  (tasks={tasks}, models={models_order})")


def fig_multi_dataset():
    """Figure set 6: Multi-dataset (legacy) comparison."""
    png = MULTI_DATASET_DIR / "multi_task_comparison.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print(
            "  [SKIP] Aggregate per-task LPAN runs into "
            f"{MULTI_DATASET_DIR}/multi_task_comparison.png"
        )


def fig_lpan():
    """Figure set 7: LPAN per-stage comparison plots."""
    lpan_dir = RESULTS_DIR / "lpan"
    pngs = sorted(lpan_dir.rglob("lpan_comparison*.png"))
    if pngs:
        for p in pngs:
            print(f"  [OK] {p}")
    else:
        print("  [SKIP] Run: python run_staged_lpan.py --model <key> --task <task>")


def fig_encrypted_inference():
    """Figure set 8: PF-SR encrypted inference latency per model and phase."""
    if not ENCRYPTED_INFERENCE_DIR.exists():
        print(
            "  [SKIP] Run: python experiments/run_protocol.py --model <key> --phase <phase>"
        )
        return

    records = []  # (model_short, phase, latency_s)
    for key, cfg in MODEL_REGISTRY.items():
        for phase in ("ffn", "attention", "layer", "model"):
            path = ENCRYPTED_INFERENCE_DIR / f"{key}_{phase}.json"
            if not path.exists():
                continue
            with path.open() as f:
                payload = json.load(f)
            latency = payload.get("latency_s") or payload.get("total_seconds") or 0.0
            records.append((cfg["short"], phase, float(latency)))

    if not records:
        print("  [SKIP] No encrypted-inference JSON in results/encrypted_inference/")
        return

    phases = ["ffn", "attention", "layer", "model"]
    models_present = sorted(
        {r[0] for r in records},
        key=lambda s: [c["short"] for c in MODEL_REGISTRY.values()].index(s),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(models_present))
    for i, phase in enumerate(phases):
        ys = [
            next((lat for (m, p, lat) in records if m == ms and p == phase), 0.0)
            for ms in models_present
        ]
        ax.bar(x + (i - 1.5) * width, ys, width, label=phase)
    ax.set_xticks(x)
    ax.set_xticklabels(models_present)
    ax.set(ylabel="Latency (s)", title="PF-SR Encrypted Inference Latency")
    ax.set_yscale("log")
    ax.legend(title="Phase")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    out = FIG_DIR / "encrypted_inference_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")


FIGURE_SETS = {
    1: ("Polynomial Approximation", fig_poly_approx),
    2: ("Activation Profiles (per model)", fig_activation_profiles),
    3: ("BSGS Evaluation", fig_bsgs),
    4: ("Error Propagation (per model)", fig_error_propagation),
    5: ("Multi-Model × GLUE Tasks", fig_multi_model),
    6: ("Multi-Dataset Comparison", fig_multi_dataset),
    7: ("LPAN Stage Comparison", fig_lpan),
    8: ("PF-SR Encrypted Inference", fig_encrypted_inference),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=int,
        nargs="*",
        help=f"Only generate specific figure sets (1-{len(FIGURE_SETS)})",
    )
    args = parser.parse_args()

    ensure_dirs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Thesis Figure Generator")
    print("=" * 70)

    sets = args.only if args.only else list(FIGURE_SETS.keys())
    for idx in sets:
        if idx in FIGURE_SETS:
            name, fn = FIGURE_SETS[idx]
            print(f"\n  [{idx}] {name}")
            fn()

    print(f"\n  Figures directory: {FIG_DIR}/")


if __name__ == "__main__":
    main()
