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
    DEPTH_ALLOCATION_DIR,
    BSGS_EVAL_DIR,
    ERROR_PROPAGATION_DIR,
    GA_OPTIMIZATION_DIR,
    MULTI_MODEL_DIR,
    MULTI_DATASET_DIR,
    ensure_dirs,
)

FIG_DIR = RESULTS_DIR / "figures"


def fig_poly_approx():
    """Figure set 1: Polynomial approximation comparisons."""
    json_path = POLY_APPROX_DIR / "comparison_results.json"
    if not json_path.exists():
        print(
            "  [SKIP] No poly_approx results found. Run: python experiments/run_analysis.py poly"
        )
        return
    with open(json_path) as f:
        data = json.load(f)
    print("  [OK] Polynomial approximation figures (run_analysis.py poly)")


def fig_activation_profiles():
    """Figure set 2: Activation distribution histograms."""
    png = ACTIVATION_PROFILES_DIR / "activation_distributions.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run: python experiments/run_analysis.py profile")


def fig_depth_allocation():
    """Figure set 3: Adaptive vs uniform depth allocation."""
    png = DEPTH_ALLOCATION_DIR / "adaptive_vs_uniform.png"
    hmap = DEPTH_ALLOCATION_DIR / "degree_heatmap.png"
    if png.exists() and hmap.exists():
        print(f"  [OK] {png}")
        print(f"  [OK] {hmap}")
    else:
        print("  [SKIP] Run experiment 02 first.")


def fig_bsgs():
    """Figure set 4: BSGS polynomial evaluation comparison."""
    png = BSGS_EVAL_DIR / "bsgs_comparison.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run: python experiments/run_analysis.py bsgs")


def fig_error_propagation():
    """Figure set 5: Error propagation analysis."""
    png = ERROR_PROPAGATION_DIR / "error_propagation.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run: python experiments/run_analysis.py error")


def fig_ga_convergence():
    """Figure set 6: GA convergence."""
    png = GA_OPTIMIZATION_DIR / "ga_convergence.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run experiment 06 first.")


def fig_multi_model():
    """Figure set 7: Multi-model scaling results."""
    json_path = MULTI_MODEL_DIR / "scaling_results.json"
    if not json_path.exists():
        print("  [SKIP] Run: python experiments/05_multi_model_scaling.py")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Recreate the scaling chart
    models = [r["short"] for r in data]
    baseline_acc = [r.get("baseline_acc", 0) * 100 for r in data]
    poly_acc = [r.get("poly_acc", 0) * 100 for r in data]
    params = [r.get("params_m", 0) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(models))
    w = 0.35
    ax1.bar(x - w / 2, baseline_acc, w, label="Baseline", color="tab:blue", alpha=0.85)
    ax1.bar(x + w / 2, poly_acc, w, label="Polynomial", color="tab:green", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set(ylabel="Accuracy (%)", title="SST-2 Accuracy: Baseline vs Polynomial")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    drops = [b - p for b, p in zip(baseline_acc, poly_acc)]
    ax2.bar(x, drops, color="tab:orange", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set(
        ylabel="Accuracy Drop (%)", title="Accuracy Drop from Polynomial Replacement"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "multi_model_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {FIG_DIR / 'multi_model_scaling.png'}")


def fig_multi_dataset():
    """Figure set 8: Multi-dataset comparison."""
    png = MULTI_DATASET_DIR / "multi_task_comparison.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run experiment 10 first.")


def fig_lpan():
    """Figure set 9: LPAN comparison."""
    lpan_dir = RESULTS_DIR / "lpan"
    png = lpan_dir / "lpan_comparison.png"
    if png.exists():
        print(f"  [OK] {png}")
    else:
        print("  [SKIP] Run: python run_staged_lpan.py --model <key>")


FIGURE_SETS = {
    1: ("Polynomial Approximation", fig_poly_approx),
    2: ("Activation Profiles", fig_activation_profiles),
    3: ("Depth Allocation", fig_depth_allocation),
    4: ("BSGS Evaluation", fig_bsgs),
    5: ("Error Propagation", fig_error_propagation),
    6: ("GA Convergence", fig_ga_convergence),
    7: ("Multi-Model Scaling", fig_multi_model),
    8: ("Multi-Dataset", fig_multi_dataset),
    9: ("LPAN Results", fig_lpan),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", type=int, nargs="*", help="Only generate specific figure sets (1-9)"
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
