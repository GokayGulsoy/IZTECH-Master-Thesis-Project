#!/usr/bin/env python3
"""
Experiment 12: LPAN-FHE Protocol — Phase 1 (FFN + LN block)
============================================================
Demonstrates the Pure-FHE Single-Round (PF-SR) protocol on a single
BERT-Tiny FFN+LN block using:

  * the `CKKSBackend` abstraction (TenSEAL impl),
  * token-packed CKKS encoding (`TokenPackedTensor`),
  * symbolic depth tracking (`DepthAudit`),
  * profile-driven LPAN polynomial coefficients (degree 8).

Inputs are encrypted *once* on the client side (a numpy array of shape
[seq_len, hidden_dim]); the server runs Linear → GELU-poly → Linear →
residual → LN-poly entirely under FHE; the client decrypts once at the
end. No MPC round-trip.

Outputs → results/encrypted_inference/phase1_protocol.json

Run on the MSI box (TenSEAL required):
    python experiments/12_lpan_fhe_protocol.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from fhe_thesis.config import (
    ENCRYPTED_INFERENCE_DIR,
    PROFILED_INTERVALS,
    ensure_dirs,
)
from fhe_thesis.encryption import (
    DepthAudit,
    TenSEALBackend,
    TokenPackedTensor,
    enc_gelu_poly,
    enc_linear,
    enc_ln_poly,
    transformer_layer_depth,
)
from fhe_thesis.encryption.depth import DEPTH_COST
from fhe_thesis.models.profiling import build_kde_density, profile_model
from fhe_thesis.poly.approximation import (
    gelu_func,
    inv_sqrt_func,
    weighted_minimax_approx,
)
from fhe_thesis.poly.chebyshev import chebyshev_to_power


MODEL = "google/bert_uncased_L-2_H-128_A-2"
SEQ_LEN = 8  # short sequence for a Phase-1 sanity benchmark
HIDDEN = 128
INTERMEDIATE = 512


# ── Plaintext reference (numpy) ────────────────────────────────────────
def plaintext_block(
    x, W1, b1, W2, b2, gelu_coeffs, gelu_iv, inv_coeffs, inv_iv, gamma, beta
):
    a, b = gelu_iv
    h1 = x @ W1.T + b1
    h1_std = (2.0 * np.clip(h1, a, b) - (a + b)) / (b - a)
    h2 = np.polynomial.polynomial.polyval(h1_std.T, gelu_coeffs).T
    h3 = h2 @ W2.T + b2
    res = h3 + x

    var = (res * res).mean(axis=-1, keepdims=True)
    a2, b2_ = inv_iv
    var_std = (2.0 * np.clip(var, a2, b2_) - (a2 + b2_)) / (b2_ - a2)
    inv_sigma = np.polynomial.polynomial.polyval(var_std.squeeze(-1), inv_coeffs)
    return gamma * (res * inv_sigma[:, None]) + beta


# ── Helper: build coefficients for one layer ───────────────────────────
def _layer_coeffs(profile_data, layer_idx, op_name, func):
    key = f"L{layer_idx}_{op_name}"
    interval = PROFILED_INTERVALS[key]
    dk = {"GELU": "gelu_inputs", "LN": "ln_variances"}[op_name]
    if dk in profile_data and layer_idx in profile_data[dk]:
        density = build_kde_density(profile_data[dk][layer_idx])
    else:
        density = lambda x: np.ones_like(x, dtype=float)
    cheb_c, _ = weighted_minimax_approx(func, interval, 8, density)
    return chebyshev_to_power(cheb_c), interval


def main():
    ensure_dirs()
    out = ENCRYPTED_INFERENCE_DIR / "phase1_protocol.json"

    print("=" * 70)
    print("  LPAN-FHE Phase 1: FFN + LN block on BERT-Tiny")
    print("=" * 70)

    # ── 1. Backend + depth audit ──────────────────────────────────────
    print("\n[1/5] Initialising CKKS backend (TenSEAL)…")
    backend = TenSEALBackend(poly_modulus_degree=16384)
    print(
        f"      slots={backend.capabilities.n_slots}, "
        f"levels={backend.capabilities.initial_levels}"
    )

    audit = DepthAudit(initial_levels=backend.capabilities.initial_levels)
    for op in ("linear", "polyval_deg8", "linear", "residual_add", "ln_poly"):
        audit.consume(op)
    print("\n[2/5] Depth audit for FFN + LN block:")
    print(audit.report())
    print(f"\n      Full layer would consume: {transformer_layer_depth()} levels")

    # ── 2. Profile + fit polynomials ──────────────────────────────────
    print("\n[3/5] Profiling activations and fitting LPAN polynomials…")
    profile = profile_model(MODEL, num_layers=2, num_samples=400)
    gelu_pc, gelu_iv = _layer_coeffs(profile, 0, "GELU", gelu_func)
    inv_pc, inv_iv = _layer_coeffs(profile, 0, "LN", inv_sqrt_func)

    # ── 3. Extract weights ────────────────────────────────────────────
    print("\n[4/5] Loading BERT-Tiny weights…")
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    sd = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    p = "bert.encoder.layer.0"
    W1 = sd[f"{p}.intermediate.dense.weight"]
    b1 = sd[f"{p}.intermediate.dense.bias"]
    W2 = sd[f"{p}.output.dense.weight"]
    b2 = sd[f"{p}.output.dense.bias"]
    gamma = sd[f"{p}.output.LayerNorm.weight"]
    beta = sd[f"{p}.output.LayerNorm.bias"]

    rng = np.random.default_rng(0)
    x_plain = rng.standard_normal((SEQ_LEN, HIDDEN)).astype(np.float64)

    # ── 4. Plaintext reference ────────────────────────────────────────
    y_ref = plaintext_block(
        x_plain, W1, b1, W2, b2, gelu_pc, gelu_iv, inv_pc, inv_iv, gamma, beta
    )

    # ── 5. Encrypted protocol run ─────────────────────────────────────
    print("\n[5/5] Running encrypted FFN+LN protocol…")
    t0 = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    t_enc = time.time() - t0

    t0 = time.time()
    h1 = enc_linear(backend, ct_x, W1, b1)
    t_lin1 = time.time() - t0

    t0 = time.time()
    h2 = enc_gelu_poly(backend, h1, gelu_pc, gelu_iv)
    t_gelu = time.time() - t0

    t0 = time.time()
    h3 = enc_linear(backend, h2, W2, b2)
    t_lin2 = time.time() - t0

    # residual: h3 + x_plain (token-wise add of two TokenPackedTensors)
    t0 = time.time()
    res_cts = [backend.add(h3.cts[i], ct_x.cts[i]) for i in range(SEQ_LEN)]
    res = TokenPackedTensor.from_ciphertexts(res_cts, hidden_dim=HIDDEN)
    t_res = time.time() - t0

    t0 = time.time()
    y_ct = enc_ln_poly(backend, res, inv_pc, inv_iv, gamma, beta)
    t_ln = time.time() - t0

    t0 = time.time()
    y_dec = y_ct.decrypt(backend)
    t_dec = time.time() - t0

    max_err = float(np.max(np.abs(y_dec - y_ref)))
    rel_err = float(np.linalg.norm(y_dec - y_ref) / (np.linalg.norm(y_ref) + 1e-12))
    total_ms = (t_enc + t_lin1 + t_gelu + t_lin2 + t_res + t_ln + t_dec) * 1000

    print(f"\n  seq_len={SEQ_LEN} hidden={HIDDEN}")
    print(f"  max |error| = {max_err:.3e}, rel error = {rel_err:.3e}")
    print(
        f"  total = {total_ms:.1f} ms  "
        f"(enc={t_enc*1e3:.1f}, lin1={t_lin1*1e3:.1f}, "
        f"gelu={t_gelu*1e3:.1f}, lin2={t_lin2*1e3:.1f}, "
        f"res={t_res*1e3:.1f}, ln={t_ln*1e3:.1f}, dec={t_dec*1e3:.1f})"
    )

    record = {
        "phase": 1,
        "block": "ffn+ln",
        "model": MODEL,
        "seq_len": SEQ_LEN,
        "hidden": HIDDEN,
        "max_abs_error": max_err,
        "rel_error": rel_err,
        "latency_ms": {
            "encrypt": t_enc * 1e3,
            "linear1": t_lin1 * 1e3,
            "gelu_poly": t_gelu * 1e3,
            "linear2": t_lin2 * 1e3,
            "residual": t_res * 1e3,
            "ln_poly": t_ln * 1e3,
            "decrypt": t_dec * 1e3,
            "total": total_ms,
        },
        "depth_consumed": audit.current,
        "depth_remaining": audit.remaining,
        "depth_per_full_layer": transformer_layer_depth(),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  ↳ saved {out}")


if __name__ == "__main__":
    main()
