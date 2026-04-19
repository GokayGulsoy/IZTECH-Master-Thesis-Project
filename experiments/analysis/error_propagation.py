#!/usr/bin/env python3
"""Error Propagation Analysis — model-agnostic.

Computes theoretical + empirical error-propagation bounds for the
LPAN polynomial-approximated Transformer layer. Iterates the
MODEL_REGISTRY so that each variant (Tiny / Mini / Small / Base)
gets its own per-layer spectral norms, per-operation polynomial
errors, layer bound, depth-amplification factor, and depth-scaling
curve.

Outputs → ``results/error_propagation/<model>/``
"""
from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial import chebyshev as cheb

from fhe_thesis.config import (
    ERROR_PROPAGATION_DIR,
    MODEL_REGISTRY,
    PROFILED_INTERVALS,
    ensure_dirs,
)
from fhe_thesis.poly.approximation import exp_func, gelu_func, inv_sqrt_func


def _interval_for(layer_idx: int, op: str):
    """Return the profiled interval for ``L{layer_idx}_{op}`` with fallback to L0."""
    key = f"L{layer_idx}_{op}"
    if key in PROFILED_INTERVALS:
        return PROFILED_INTERVALS[key]
    return PROFILED_INTERVALS[f"L0_{op}"]


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
        return self.attention_error_bound(
            eps_sm, eps_ln, sigma_ckks, seq_len
        ) + self.ffn_error_bound(eps_gelu, eps_ln, sigma_ckks)

    def multi_layer_bound(self, per_layer_errors, alpha=1.0):
        L = len(per_layer_errors)
        if abs(alpha - 1.0) < 1e-10:
            return sum(per_layer_errors)
        return sum(alpha ** (L - 1 - i) * e for i, e in enumerate(per_layer_errors))


def compute_spectral_norms(model_path: str, num_layers: int):
    """Compute spectral norms of every encoder-layer weight matrix."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    result = {}
    for layer_idx in range(num_layers):
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


def _analyse_one_model(
    model_key: str, deg_gelu: int, deg_softmax: int, deg_ln: int
) -> None:
    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["name"]
    num_layers = cfg["layers"]
    out = ERROR_PROPAGATION_DIR / model_key
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"\n  → {cfg['short']} ({model_name}, L={num_layers}, d={cfg['hidden']}, H={cfg['heads']})"
    )

    print("    [1/4] Computing spectral norms...")
    spectral_norms = compute_spectral_norms(model_name, num_layers)

    bounds = ErrorBounds(
        {
            "hidden_size": cfg["hidden"],
            "intermediate_size": cfg["hidden"] * 4,
            "num_attention_heads": cfg["heads"],
            "num_hidden_layers": num_layers,
        }
    )

    print("    [2/4] Per-operation polynomial errors...")
    degrees_per_op: dict = {}
    op_errors: dict = {}
    for layer_idx in range(num_layers):
        for op_name, deg in [
            ("GELU", deg_gelu),
            ("Softmax", deg_softmax),
            ("LN", deg_ln),
        ]:
            key = f"L{layer_idx}_{op_name}"
            interval = _interval_for(layer_idx, op_name)
            eps = bounds.poly_approx_error(op_name, deg, interval)
            degrees_per_op[key] = deg
            op_errors[key] = eps

    print("    [3/4] Per-layer + model bounds...")
    layer_errors = []
    for layer_idx in range(num_layers):
        for k_s, attr_s in [
            ("Wq", "sigma_Wq"),
            ("Wk", "sigma_Wk"),
            ("Wv", "sigma_Wv"),
            ("Wo", "sigma_Wo"),
            ("W1", "sigma_W1"),
            ("W2", "sigma_W2"),
        ]:
            setattr(bounds, attr_s, spectral_norms.get(f"L{layer_idx}_{k_s}", 1.0))

        layer_err = bounds.layer_error_bound(
            op_errors[f"L{layer_idx}_GELU"],
            op_errors[f"L{layer_idx}_Softmax"],
            op_errors[f"L{layer_idx}_LN"],
        )
        layer_errors.append(layer_err)

    alpha = 1.0 + max(
        spectral_norms.get(f"L{i}_{k}", 1.0)
        for i in range(num_layers)
        for k in ("Wo", "W2")
    )
    total = bounds.multi_layer_bound(layer_errors, alpha)
    print(f"    layer bounds = {[f'{e:.2e}' for e in layer_errors]}")
    print(f"    amplification α = {alpha:.3f}  →  total bound = {total:.3e}")

    depth_scaling: dict = {}
    avg_err = float(np.mean(layer_errors))
    for L in [2, 4, 6, 8, 12, 24]:
        if abs(alpha - 1.0) < 1e-10:
            depth_scaling[L] = avg_err * L
        else:
            depth_scaling[L] = avg_err * (alpha**L - 1) / (alpha - 1)

    print("    [4/4] Degree sweep + plots...")
    sweep_intervals = {op: _interval_for(0, op) for op in ("GELU", "Softmax", "LN")}
    sweep_degrees = list(range(2, 17, 2))
    sweep_results = {
        op: {d: bounds.poly_approx_error(op, d, iv) for d in sweep_degrees}
        for op, iv in sweep_intervals.items()
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for op_name, color in [
        ("GELU", "tab:blue"),
        ("Softmax", "tab:red"),
        ("LN", "tab:green"),
    ]:
        errs = [sweep_results[op_name][d] for d in sweep_degrees]
        ax1.semilogy(sweep_degrees, errs, "o-", label=op_name, color=color, linewidth=2)
    ax1.set(
        xlabel="Polynomial Degree",
        ylabel="Max Absolute Error",
        title=f"{cfg['short']}: Approximation Error vs Degree",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    Ls = sorted(depth_scaling.keys())
    ax2.semilogy(
        Ls, [depth_scaling[l] for l in Ls], "D-", color="tab:purple", linewidth=2
    )
    ax2.set(
        xlabel="Number of Layers",
        ylabel="Total Error Bound",
        title=f"{cfg['short']}: Error Growth with Depth (α={alpha:.2f})",
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "error_propagation.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "model": cfg["short"],
        "model_path": model_name,
        "config": {
            "layers": num_layers,
            "hidden": cfg["hidden"],
            "heads": cfg["heads"],
        },
        "degrees": degrees_per_op,
        "per_operation_errors": {k: float(v) for k, v in op_errors.items()},
        "per_layer_bounds": {f"L{i}": float(e) for i, e in enumerate(layer_errors)},
        "amplification_factor": alpha,
        "total_error_bound": total,
        "spectral_norms": spectral_norms,
        "depth_scaling": {str(k): float(v) for k, v in depth_scaling.items()},
        "degree_sweep": {
            op: {str(d): float(e) for d, e in errs.items()}
            for op, errs in sweep_results.items()
        },
    }
    with (out / "error_analysis.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"    saved → {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        nargs="*",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY),
    )
    parser.add_argument("--deg-gelu", type=int, default=8)
    parser.add_argument("--deg-softmax", type=int, default=8)
    parser.add_argument("--deg-ln", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs()
    print("=" * 70)
    print("  Error Propagation Analysis — multi-model")
    print("=" * 70)
    for key in args.model:
        _analyse_one_model(key, args.deg_gelu, args.deg_softmax, args.deg_ln)
    print(f"\n  All results saved under {ERROR_PROPAGATION_DIR}/")


if __name__ == "__main__":
    main()
