#!/usr/bin/env python3
"""
Experiment 13: LPAN-FHE Protocol — Phase 2 (encrypted self-attention)
======================================================================
Demonstrates fully-encrypted multi-head self-attention on BERT-Tiny:

  * Q/K/V plaintext-weight projections (per head)
  * Q·Kᵀ scaled-dot-product scores under FHE
  * LPAN softmax-polynomial applied row-wise
  * attention·V under FHE
  * head concatenation by zero-padding masks (no decryption)
  * output projection

No client interaction at any point — the PF-SR protocol invariant
holds end-to-end.

Outputs → results/encrypted_inference/phase2_protocol.json

Run on the MSI box (TenSEAL required):
    python experiments/13_lpan_fhe_attention.py
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
    enc_self_attention,
    transformer_layer_depth,
)
from fhe_thesis.models.profiling import build_kde_density, profile_model
from fhe_thesis.poly.approximation import (
    exp_func,
    weighted_minimax_approx,
)
from fhe_thesis.poly.chebyshev import chebyshev_to_power


MODEL = "google/bert_uncased_L-2_H-128_A-2"
SEQ_LEN = 8  # short for a Phase-2 sanity benchmark
HIDDEN = 128
NUM_HEADS = 2


# ── Plaintext reference (numpy) ────────────────────────────────────────
def plaintext_attention(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, sm_coeffs, sm_iv, num_heads):
    head_dim = x.shape[-1] // num_heads
    inv_d = 1.0 / np.sqrt(head_dim)
    L = x.shape[0]

    head_outs = []
    for h in range(num_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        Q = x @ Wq[s:e, :].T + bq[s:e]
        K = x @ Wk[s:e, :].T + bk[s:e]
        V = x @ Wv[s:e, :].T + bv[s:e]
        S = inv_d * (Q @ K.T)  # (L, L)
        a, b = sm_iv
        S_std = (2.0 * np.clip(S, a, b) - (a + b)) / (b - a)
        A = np.polynomial.polynomial.polyval(S_std.T, sm_coeffs).T
        head_outs.append(A @ V)
    concat = np.concatenate(head_outs, axis=-1)
    return concat @ Wo.T + bo


# ── Fit softmax polynomial from profiled inputs ────────────────────────
def _softmax_coeffs(profile_data, layer_idx=0):
    key = f"L{layer_idx}_Softmax"
    interval = PROFILED_INTERVALS[key]
    if "softmax_inputs" in profile_data and layer_idx in profile_data["softmax_inputs"]:
        density = build_kde_density(profile_data["softmax_inputs"][layer_idx])
    else:
        density = lambda x: np.ones_like(x, dtype=float)
    cheb_c, _ = weighted_minimax_approx(exp_func, interval, 8, density)
    return chebyshev_to_power(cheb_c), interval


def main():
    ensure_dirs()
    out = ENCRYPTED_INFERENCE_DIR / "phase2_protocol.json"

    print("=" * 70)
    print("  LPAN-FHE Phase 2: Multi-head self-attention on BERT-Tiny")
    print("=" * 70)

    # ── 1. Backend ────────────────────────────────────────────────────
    print("\n[1/5] Initialising CKKS backend (TenSEAL, N=16384)…")
    backend = TenSEALBackend(poly_modulus_degree=16384)
    print(
        f"      slots={backend.capabilities.n_slots}, "
        f"levels={backend.capabilities.initial_levels}"
    )

    audit = DepthAudit(initial_levels=backend.capabilities.initial_levels)
    for op in ("linear", "qk_scores", "softmax_poly", "attn_apply",
               "head_concat", "linear"):
        audit.consume(op)
    print("\n[2/5] Depth audit for self-attention block:")
    print(audit.report())
    print(f"\n      Full LPAN layer would consume: {transformer_layer_depth()} levels")

    # ── 2. Profile + softmax-poly fit ─────────────────────────────────
    print("\n[3/5] Profiling activations and fitting softmax polynomial…")
    profile = profile_model(MODEL, num_layers=2, num_samples=400)
    sm_pc, sm_iv = _softmax_coeffs(profile, layer_idx=0)

    # ── 3. Weights ────────────────────────────────────────────────────
    print("\n[4/5] Loading BERT-Tiny attention weights…")
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    sd = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    p = "bert.encoder.layer.0.attention"
    Wq = sd[f"{p}.self.query.weight"]
    bq = sd[f"{p}.self.query.bias"]
    Wk = sd[f"{p}.self.key.weight"]
    bk = sd[f"{p}.self.key.bias"]
    Wv = sd[f"{p}.self.value.weight"]
    bv = sd[f"{p}.self.value.bias"]
    Wo = sd[f"{p}.output.dense.weight"]
    bo = sd[f"{p}.output.dense.bias"]

    rng = np.random.default_rng(0)
    x_plain = rng.standard_normal((SEQ_LEN, HIDDEN)).astype(np.float64) * 0.1

    # ── 4. Plaintext reference ────────────────────────────────────────
    y_ref = plaintext_attention(
        x_plain,
        Wq,
        bq,
        Wk,
        bk,
        Wv,
        bv,
        Wo,
        bo,
        sm_pc,
        sm_iv,
        NUM_HEADS,
    )

    # ── 5. Encrypted protocol run ─────────────────────────────────────
    print("\n[5/5] Running encrypted self-attention…")
    t0 = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    t_enc = time.time() - t0

    t0 = time.time()
    y_ct = enc_self_attention(
        backend,
        ct_x,
        Wq,
        bq,
        Wk,
        bk,
        Wv,
        bv,
        Wo,
        bo,
        softmax_power_coeffs=sm_pc,
        softmax_interval=sm_iv,
        num_heads=NUM_HEADS,
    )
    t_attn = time.time() - t0

    t0 = time.time()
    y_dec = y_ct.decrypt(backend)
    t_dec = time.time() - t0

    max_err = float(np.max(np.abs(y_dec - y_ref)))
    rel_err = float(np.linalg.norm(y_dec - y_ref) / (np.linalg.norm(y_ref) + 1e-12))
    total_ms = (t_enc + t_attn + t_dec) * 1000

    print(f"\n  seq_len={SEQ_LEN} hidden={HIDDEN} heads={NUM_HEADS}")
    print(f"  max |error| = {max_err:.3e}, rel error = {rel_err:.3e}")
    print(
        f"  total = {total_ms:.1f} ms  "
        f"(enc={t_enc*1e3:.1f}, attn={t_attn*1e3:.1f}, dec={t_dec*1e3:.1f})"
    )

    record = {
        "phase": 2,
        "block": "self_attention",
        "model": MODEL,
        "seq_len": SEQ_LEN,
        "hidden": HIDDEN,
        "num_heads": NUM_HEADS,
        "max_abs_error": max_err,
        "rel_error": rel_err,
        "latency_ms": {
            "encrypt": t_enc * 1e3,
            "attention": t_attn * 1e3,
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
