#!/usr/bin/env python3
"""
Experiment 03: Activation Distribution Profiling
==================================================
Profiles GELU, Softmax, and LayerNorm activation inputs on BERT-Tiny
using SST-2, saves histograms, statistics, and fitted KDE densities.

Outputs → results/activation_profiles/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fhe_thesis.config import ACTIVATION_PROFILES_DIR, ensure_dirs
from fhe_thesis.models.profiling import profile_model, build_kde_density

OUT = ACTIVATION_PROFILES_DIR


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Activation Distribution Profiling — BERT-Tiny")
    print("=" * 70)

    model_name = "google/bert_uncased_L-2_H-128_A-2"
    num_layers = 2

    print("\n[1/3] Running profiling inference...")
    profiles = profile_model(model_name, num_layers, num_samples=2000)

    # Summary statistics
    print("\n[2/3] Computing statistics and saving data...")
    stats = {}
    for act_type in ["gelu_inputs", "softmax_inputs", "ln_variances"]:
        stats[act_type] = {}
        for layer_idx in sorted(profiles[act_type].keys()):
            data = profiles[act_type][layer_idx]
            s = {
                "count": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "p0.5": float(np.percentile(data, 0.5)),
                "p1": float(np.percentile(data, 1)),
                "p5": float(np.percentile(data, 5)),
                "p25": float(np.percentile(data, 25)),
                "p50": float(np.percentile(data, 50)),
                "p75": float(np.percentile(data, 75)),
                "p95": float(np.percentile(data, 95)),
                "p99": float(np.percentile(data, 99)),
                "p99.5": float(np.percentile(data, 99.5)),
                "recommended_interval": [
                    float(np.percentile(data, 0.5)),
                    float(np.percentile(data, 99.5)),
                ],
            }
            stats[act_type][str(layer_idx)] = s
            print(f"  {act_type} L{layer_idx}: n={s['count']}, "
                  f"mean={s['mean']:.3f}, std={s['std']:.3f}, "
                  f"interval=[{s['p0.5']:.3f}, {s['p99.5']:.3f}]")

            # Save raw samples (sub-sampled to 50K)
            if len(data) > 50000:
                rng = np.random.default_rng(42)
                data = rng.choice(data, 50000, replace=False)
            np.save(OUT / f"{act_type}_layer{layer_idx}.npy", data)

    with open(OUT / "profile_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plots
    print("\n[3/3] Generating distribution plots...")
    act_labels = {
        "gelu_inputs": "GELU Input",
        "softmax_inputs": "Softmax Input (QK^T/√d_k)",
        "ln_variances": "LayerNorm Variance",
    }

    fig, axes = plt.subplots(num_layers, 3, figsize=(16, 4 * num_layers))
    if num_layers == 1:
        axes = axes[np.newaxis, :]

    for row, layer_idx in enumerate(range(num_layers)):
        for col, act_type in enumerate(["gelu_inputs", "softmax_inputs", "ln_variances"]):
            ax = axes[row, col]
            data = profiles[act_type][layer_idx]
            ax.hist(data, bins=200, density=True, alpha=0.7, color="steelblue")

            # Overlay KDE
            density = build_kde_density(data)
            lo, hi = np.percentile(data, [0.1, 99.9])
            x_kde = np.linspace(lo, hi, 500)
            ax.plot(x_kde, density(x_kde), "r-", linewidth=1.5, label="KDE")

            # Mark recommended interval
            p05, p995 = np.percentile(data, 0.5), np.percentile(data, 99.5)
            ax.axvline(p05, color="green", linestyle="--", alpha=0.7, label=f"[{p05:.1f}, {p995:.1f}]")
            ax.axvline(p995, color="green", linestyle="--", alpha=0.7)

            ax.set_title(f"Layer {layer_idx}: {act_labels[act_type]}", fontsize=10)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

    plt.suptitle("BERT-Tiny Activation Distributions (SST-2)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "activation_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved activation_distributions.png")

    print(f"\n  All results saved to: {OUT}/")


if __name__ == "__main__":
    main()
