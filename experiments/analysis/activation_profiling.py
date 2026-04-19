#!/usr/bin/env python3
"""Activation Distribution Profiling — model-agnostic.

Profiles GELU, Softmax, and LayerNorm activation inputs on the chosen
BERT variant(s) using SST-2, saves histograms, statistics, and fitted
KDE densities. Used to derive the per-layer approximation intervals
that ``fhe_thesis.config.PROFILED_INTERVALS`` and the polynomial fits
in ``fhe_thesis.encryption.coefficients`` rely on.

Outputs → ``results/activation_profiles/<model>/``
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fhe_thesis.config import (
    ACTIVATION_PROFILES_DIR,
    MODEL_REGISTRY,
    ensure_dirs,
)
from fhe_thesis.models.profiling import build_kde_density, profile_model


ACT_LABELS = {
    "gelu_inputs": "GELU Input",
    "softmax_inputs": "Softmax Input (QK^T/√d_k)",
    "ln_variances": "LayerNorm Variance",
}


def _profile_one_model(model_key: str, num_samples: int) -> None:
    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["name"]
    num_layers = cfg["layers"]

    out = ACTIVATION_PROFILES_DIR / model_key
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  → {cfg['short']} ({model_name}, {num_layers} layers)")
    profiles = profile_model(model_name, num_layers, num_samples=num_samples)

    stats: dict = {}
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
                **{
                    f"p{p}": float(np.percentile(data, p))
                    for p in (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5)
                },
                "recommended_interval": [
                    float(np.percentile(data, 0.5)),
                    float(np.percentile(data, 99.5)),
                ],
            }
            stats[act_type][str(layer_idx)] = s
            print(
                f"    {act_type:>14s} L{layer_idx}: n={s['count']}, "
                f"mean={s['mean']:.3f}, std={s['std']:.3f}, "
                f"interval=[{s['p0.5']:.3f}, {s['p99.5']:.3f}]"
            )

            if len(data) > 50000:
                rng = np.random.default_rng(42)
                data = rng.choice(data, 50000, replace=False)
            np.save(out / f"{act_type}_layer{layer_idx}.npy", data)

    with (out / "profile_statistics.json").open("w") as f:
        json.dump(stats, f, indent=2)

    fig, axes = plt.subplots(num_layers, 3, figsize=(16, 4 * num_layers))
    if num_layers == 1:
        axes = axes[np.newaxis, :]

    for row, layer_idx in enumerate(range(num_layers)):
        for col, act_type in enumerate(
            ["gelu_inputs", "softmax_inputs", "ln_variances"]
        ):
            ax = axes[row, col]
            data = profiles[act_type][layer_idx]
            ax.hist(data, bins=200, density=True, alpha=0.7, color="steelblue")

            density = build_kde_density(data)
            lo, hi = np.percentile(data, [0.1, 99.9])
            x_kde = np.linspace(lo, hi, 500)
            ax.plot(x_kde, density(x_kde), "r-", linewidth=1.5, label="KDE")

            p05, p995 = np.percentile(data, [0.5, 99.5])
            ax.axvline(
                p05,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"[{p05:.1f}, {p995:.1f}]",
            )
            ax.axvline(p995, color="green", linestyle="--", alpha=0.7)

            ax.set_title(f"Layer {layer_idx}: {ACT_LABELS[act_type]}", fontsize=10)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

    plt.suptitle(
        f"{cfg['short']} Activation Distributions (SST-2)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out / "activation_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved → {out}/activation_distributions.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        nargs="*",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY),
        help="Which model variants to profile (default: all)",
    )
    parser.add_argument("--num-samples", type=int, default=2000)
    args = parser.parse_args()

    ensure_dirs()
    print("=" * 70)
    print("  Activation Distribution Profiling — multi-model")
    print("=" * 70)
    print(f"  models = {args.model}, samples = {args.num_samples}")

    for key in args.model:
        _profile_one_model(key, args.num_samples)

    print(f"\n  All results saved under {ACTIVATION_PROFILES_DIR}/")


if __name__ == "__main__":
    main()
