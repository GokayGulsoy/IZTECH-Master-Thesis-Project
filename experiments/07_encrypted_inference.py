#!/usr/bin/env python3
"""
Experiment 07: CKKS Encrypted Inference (Contribution 4)
=========================================================
Benchmarks encrypted FFN, self-attention, and full Transformer layer
on BERT-Tiny using TenSEAL CKKS scheme.

Outputs → results/encrypted_inference/
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import tenseal as ts
from numpy.polynomial.chebyshev import chebval

from fhe_thesis.config import (
    PROFILED_INTERVALS, ENCRYPTED_INFERENCE_DIR, ensure_dirs,
)
from fhe_thesis.poly.approximation import (
    gelu_func, exp_func, inv_sqrt_func, weighted_minimax_approx,
)
from fhe_thesis.poly.chebyshev import chebyshev_to_power
from fhe_thesis.encryption.context import create_ckks_context
from fhe_thesis.models.profiling import build_kde_density, profile_model

OUT = ENCRYPTED_INFERENCE_DIR


def enc_linear(enc_x, weight, bias):
    """Encrypted linear: enc_x @ W^T + b."""
    enc_out = enc_x.matmul(weight.T.tolist())
    enc_out = enc_out + bias.tolist()
    return enc_out


def enc_polynomial(enc_x, cheb_coeffs, interval):
    """Evaluate Chebyshev polynomial on encrypted data."""
    a, b = interval
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    enc_std = enc_x * scale + shift
    power_coeffs = chebyshev_to_power(cheb_coeffs)
    return enc_std.polyval(power_coeffs.tolist())


def load_poly_coefficients(profile_data):
    """Build polynomial coefficients from profiled densities."""
    func_map = {"GELU": gelu_func, "Softmax": exp_func, "LN": inv_sqrt_func}
    density_keys = {"GELU": "gelu_inputs", "Softmax": "softmax_inputs", "LN": "ln_variances"}

    result = {}
    for layer_idx in range(2):
        for op_name in ["GELU", "Softmax", "LN"]:
            key = f"L{layer_idx}_{op_name}"
            interval = PROFILED_INTERVALS[key]
            dk = density_keys[op_name]
            if dk in profile_data and layer_idx in profile_data[dk]:
                density = build_kde_density(profile_data[dk][layer_idx])
            else:
                density = lambda x: np.ones_like(x, dtype=float)
            cheb_c, _ = weighted_minimax_approx(func_map[op_name], interval, 8, density)
            power_c = chebyshev_to_power(cheb_c)
            result[key] = {
                "cheb_coeffs": cheb_c, "power_coeffs": power_c,
                "interval": interval, "degree": 8,
            }
    return result


def extract_weights(model_path="google/bert_uncased_L-2_H-128_A-2"):
    """Extract weight matrices from a BERT model."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    return weights


def benchmark_ffn(ctx, weights, poly_coeffs, layer_idx=0):
    """Benchmark encrypted FFN block."""
    print(f"\n{'─'*60}")
    print(f"  Benchmark: Encrypted FFN Block (Layer {layer_idx})")
    print(f"{'─'*60}")

    hidden_size = 128
    intermediate_size = 512
    prefix = f"bert.encoder.layer.{layer_idx}"

    W1 = weights[f"{prefix}.intermediate.dense.weight"]
    b1 = weights[f"{prefix}.intermediate.dense.bias"]
    W2 = weights[f"{prefix}.output.dense.weight"]
    b2 = weights[f"{prefix}.output.dense.bias"]
    gelu_pc = poly_coeffs[f"L{layer_idx}_GELU"]

    np.random.seed(42)
    x_plain = np.random.randn(hidden_size).astype(np.float64)

    # Plaintext reference
    h1 = W1 @ x_plain + b1
    a, b = gelu_pc["interval"]
    h1_std = (2.0 * np.clip(h1, a, b) - (a + b)) / (b - a)
    h2 = chebval(h1_std, gelu_pc["cheb_coeffs"])
    h3 = W2 @ h2 + b2
    h_ref = h3 + x_plain

    # Encrypted
    t_start = time.time()
    enc_x = ts.ckks_vector(ctx, x_plain.tolist())
    t_enc = time.time() - t_start

    chunk_size = 128
    t0 = time.time()
    enc_chunks = []
    for i in range(0, intermediate_size, chunk_size):
        enc_chunks.append(enc_linear(enc_x, W1[i:i+chunk_size], b1[i:i+chunk_size]))
    t_lin1 = time.time() - t0

    t0 = time.time()
    enc_gelu_chunks = [enc_polynomial(ec, gelu_pc["cheb_coeffs"], gelu_pc["interval"])
                       for ec in enc_chunks]
    t_gelu = time.time() - t0

    t0 = time.time()
    enc_out = None
    for i, ec in enumerate(enc_gelu_chunks):
        partial = ec.matmul(W2[:, i*chunk_size:(i+1)*chunk_size].T.tolist())
        enc_out = partial if enc_out is None else enc_out + partial
    enc_out = enc_out + b2.tolist()
    t_lin2 = time.time() - t0

    t0 = time.time()
    enc_res = enc_out + enc_x
    t_res = time.time() - t0

    t0 = time.time()
    dec = np.array(enc_res.decrypt())
    t_dec = time.time() - t0

    max_err = float(np.max(np.abs(dec[:hidden_size] - h_ref)))
    total = (t_enc + t_lin1 + t_gelu + t_lin2 + t_res + t_dec) * 1000

    print(f"  Max |error|: {max_err:.2e}")
    print(f"  Total time:  {total:.1f} ms")
    print(f"    Encrypt={t_enc*1000:.1f} Lin1={t_lin1*1000:.1f} "
          f"GELU={t_gelu*1000:.1f} Lin2={t_lin2*1000:.1f} "
          f"Res={t_res*1000:.1f} Dec={t_dec*1000:.1f}")

    return {
        "benchmark": "FFN", "layer": layer_idx,
        "max_error": max_err, "total_time_ms": total,
        "encrypt_ms": t_enc*1000, "linear1_ms": t_lin1*1000,
        "poly_gelu_ms": t_gelu*1000, "linear2_ms": t_lin2*1000,
        "decrypt_ms": t_dec*1000,
    }


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CKKS Encrypted Inference Benchmarks — BERT-Tiny")
    print("=" * 70)

    print("\n[1/4] Profiling activations for poly coefficients...")
    profile_data = profile_model(
        "google/bert_uncased_L-2_H-128_A-2", num_layers=2, num_samples=500,
    )
    poly_coeffs = load_poly_coefficients(profile_data)

    print("\n[2/4] Loading model weights...")
    weights = extract_weights()

    print("\n[3/4] Creating CKKS context...")
    ctx = create_ckks_context()

    print("\n[4/4] Running benchmarks...")
    results = []
    for layer_idx in range(2):
        r = benchmark_ffn(ctx, weights, poly_coeffs, layer_idx)
        results.append(r)

    with open(OUT / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {OUT}/")


if __name__ == "__main__":
    main()
