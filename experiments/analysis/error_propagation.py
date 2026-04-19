#!/usr/bin/env python3
"""
Experiment 08: Error Propagation Analysis (Contribution 5)
============================================================
Computes theoretical + empirical error propagation bounds for
polynomial-approximated Transformer layers.

Outputs → results/error_propagation/
"""
from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial import chebyshev as cheb
from scipy.special import erf

from fhe_thesis.config import (
    PROFILED_INTERVALS, ERROR_PROPAGATION_DIR, ensure_dirs,
)
from fhe_thesis.poly.approximation import gelu_func, exp_func, inv_sqrt_func

OUT = ERROR_PROPAGATION_DIR


class ErrorBounds:
    """Compute theoretical error bounds for polynomial-approximated Transformer."""

    LIPSCHITZ = {"GELU": 1.5, "Softmax": 8.0, "LN": 0.7}

    def __init__(self, model_config):
        self.d_model = model_config.get("hidden_size", 128)
        self.d_ff = model_config.get("intermediate_size", 512)
        self.n_heads = model_config.get("num_attention_heads", 2)
        self.sigma_Wq = model_config.get("sigma_Wq", 1.0)
        self.sigma_Wk = model_config.get("sigma_Wk", 1.0)
        self.sigma_Wv = model_config.get("sigma_Wv", 1.0)
        self.sigma_Wo = model_config.get("sigma_Wo", 1.0)
        self.sigma_W1 = model_config.get("sigma_W1", 1.0)
        self.sigma_W2 = model_config.get("sigma_W2", 1.0)

    def poly_approx_error(self, func_name, degree, interval, n_test=10000):
        func_map_ = {"GELU": gelu_func, "Softmax": exp_func, "LN": inv_sqrt_func}
        func = func_map_[func_name]
        a, b = interval
        k = np.arange(degree + 1)
        cheb_nodes = np.cos(np.pi * k / degree)
        x_nodes = 0.5 * ((b - a) * cheb_nodes + (a + b))
        y_nodes = func(x_nodes)
        coeffs = cheb.chebfit(cheb_nodes, y_nodes, degree)
        x_test = np.linspace(a, b, n_test)
        x_test_cheb = (2 * x_test - (a + b)) / (b - a)
        y_true = func(x_test)
        y_poly = cheb.chebval(x_test_cheb, coeffs)
        return float(np.max(np.abs(y_true - y_poly)))

    def attention_error_bound(self, eps_sm, eps_ln, sigma_ckks=0.0, seq_len=128):
        x_norm = math.sqrt(self.d_model)
        attn_weight_error = seq_len * eps_sm
        attn_output_error = self.sigma_Wo * attn_weight_error * self.sigma_Wv * x_norm
        ln_error = eps_ln * x_norm
        ckks_noise = 6 * sigma_ckks * math.sqrt(self.d_model) if sigma_ckks > 0 else 0
        return attn_output_error + ln_error + ckks_noise

    def ffn_error_bound(self, eps_gelu, eps_ln, sigma_ckks=0.0):
        gelu_contrib = self.sigma_W2 * math.sqrt(self.d_ff) * eps_gelu
        ln_contrib = eps_ln * math.sqrt(self.d_model)
        ckks_noise = 4 * sigma_ckks * math.sqrt(self.d_model) if sigma_ckks > 0 else 0
        return gelu_contrib + ln_contrib + ckks_noise

    def layer_error_bound(self, eps_gelu, eps_sm, eps_ln, sigma_ckks=0.0, seq_len=128):
        return (self.attention_error_bound(eps_sm, eps_ln, sigma_ckks, seq_len)
                + self.ffn_error_bound(eps_gelu, eps_ln, sigma_ckks))

    def multi_layer_bound(self, per_layer_errors, alpha=1.0):
        L = len(per_layer_errors)
        if abs(alpha - 1.0) < 1e-10:
            return sum(per_layer_errors)
        return sum(alpha ** (L - 1 - i) * e for i, e in enumerate(per_layer_errors))


def compute_spectral_norms(model_path="google/bert_uncased_L-2_H-128_A-2"):
    """Compute spectral norms of weight matrices."""
    from transformers import AutoModelForSequenceClassification
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2", num_labels=2)

    result = {}
    for layer_idx in range(2):
        prefix = f"bert.encoder.layer.{layer_idx}"
        for name_suffix, key_suffix in [
            (".attention.self.query.weight", "Wq"),
            (".attention.self.key.weight", "Wk"),
            (".attention.self.value.weight", "Wv"),
            (".attention.output.dense.weight", "Wo"),
            (".intermediate.dense.weight", "W1"),
            (".output.dense.weight", "W2"),
        ]:
            full_name = prefix + name_suffix
            param = dict(model.named_parameters()).get(full_name)
            if param is not None:
                with torch.no_grad():
                    _, S, _ = torch.linalg.svd(param.float(), full_matrices=False)
                    result[f"L{layer_idx}_{key_suffix}"] = float(S[0])
            else:
                result[f"L{layer_idx}_{key_suffix}"] = 1.0
    return result


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Error Propagation Analysis (Contribution 5)")
    print("=" * 70)

    degrees_per_op = {
        "L0_GELU": 8, "L0_Softmax": 8, "L0_LN": 4,
        "L1_GELU": 8, "L1_Softmax": 8, "L1_LN": 2,
    }

    print("\n[1/4] Computing spectral norms...")
    spectral_norms = compute_spectral_norms()
    for k, v in spectral_norms.items():
        print(f"  {k}: {v:.4f}")

    print("\n[2/4] Computing per-operation errors...")
    bounds = ErrorBounds({
        "hidden_size": 128, "intermediate_size": 512,
        "num_attention_heads": 2, "num_hidden_layers": 2,
        **{f"sigma_{k.split('_')[1]}": v for k, v in spectral_norms.items() if k.startswith("L0")},
    })

    op_errors = {}
    for key, deg in degrees_per_op.items():
        op_name = key[3:]
        interval = PROFILED_INTERVALS[key]
        eps = bounds.poly_approx_error(op_name, deg, interval)
        op_errors[key] = eps
        print(f"  {key}: deg={deg}, ε={eps:.6e}")

    print("\n[3/4] Computing layer + model error bounds...")
    layer_errors = []
    for layer_idx in range(2):
        for k_s, attr_s in [("Wq", "sigma_Wq"), ("Wk", "sigma_Wk"),
                            ("Wv", "sigma_Wv"), ("Wo", "sigma_Wo"),
                            ("W1", "sigma_W1"), ("W2", "sigma_W2")]:
            setattr(bounds, attr_s, spectral_norms.get(f"L{layer_idx}_{k_s}", 1.0))

        layer_err = bounds.layer_error_bound(
            op_errors[f"L{layer_idx}_GELU"],
            op_errors[f"L{layer_idx}_Softmax"],
            op_errors[f"L{layer_idx}_LN"],
        )
        layer_errors.append(layer_err)
        print(f"  Layer {layer_idx} error bound: {layer_err:.6e}")

    alpha = 1.0 + max(
        spectral_norms.get("L0_Wo", 1.0), spectral_norms.get("L0_W2", 1.0),
        spectral_norms.get("L1_Wo", 1.0), spectral_norms.get("L1_W2", 1.0),
    )
    total = bounds.multi_layer_bound(layer_errors, alpha)
    print(f"  Total model bound (alpha={alpha:.2f}): {total:.6e}")

    # Depth scaling
    depth_scaling = {}
    avg_err = np.mean(layer_errors)
    for L in [2, 4, 6, 8, 12, 24]:
        if abs(alpha - 1.0) < 1e-10:
            depth_scaling[L] = avg_err * L
        else:
            depth_scaling[L] = avg_err * (alpha**L - 1) / (alpha - 1)
        print(f"  L={L:2d}: {depth_scaling[L]:.6e}")

    # Degree sweep
    print("\n[4/4] Degree sweep analysis + plots...")
    intervals = {"GELU": (-6.365, 4.173), "Softmax": (-2.101, 4.523), "LN": (0.905, 6.714)}
    sweep_degrees = list(range(2, 17, 2))
    sweep_results = {}
    for op_name, interval in intervals.items():
        sweep_results[op_name] = {}
        for deg in sweep_degrees:
            sweep_results[op_name][deg] = bounds.poly_approx_error(op_name, deg, interval)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for op_name, color in [("GELU", "tab:blue"), ("Softmax", "tab:red"), ("LN", "tab:green")]:
        errs = [sweep_results[op_name][d] for d in sweep_degrees]
        ax1.semilogy(sweep_degrees, errs, "o-", label=op_name, color=color, linewidth=2)
    ax1.set(xlabel="Polynomial Degree", ylabel="Max Absolute Error",
            title="Approximation Error vs Degree")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    Ls = sorted(depth_scaling.keys())
    ax2.semilogy(Ls, [depth_scaling[l] for l in Ls], "D-", color="tab:purple", linewidth=2)
    ax2.set(xlabel="Number of Layers", ylabel="Total Error Bound",
            title=f"Error Growth with Depth (α={alpha:.2f})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "error_propagation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved error_propagation.png")

    results = {
        "degrees": degrees_per_op,
        "per_operation_errors": {k: float(v) for k, v in op_errors.items()},
        "per_layer_bounds": {f"L{i}": float(e) for i, e in enumerate(layer_errors)},
        "amplification_factor": alpha,
        "total_error_bound": total,
        "spectral_norms": spectral_norms,
        "depth_scaling": {str(k): float(v) for k, v in depth_scaling.items()},
        "degree_sweep": {op: {str(d): float(e) for d, e in errs.items()}
                        for op, errs in sweep_results.items()},
    }
    with open(OUT / "error_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  All results saved to: {OUT}/")


if __name__ == "__main__":
    main()
