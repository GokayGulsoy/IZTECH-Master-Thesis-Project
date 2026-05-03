"""Estimate e2e BERT-tiny / BERT-base time using nexus_linear timings.

Walks through every linear in the model and times nexus_linear vs the
production matmul_plain on the actual shape. Reports a back-of-envelope
e2e projection.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


# (in_dim, out_dim, count_per_layer, num_layers, name)
BERT_TINY = [
    (128, 128, 4, 2, "attn_qkvo"),  # 4 linears: Wq, Wk, Wv, Wo
    (128, 512, 1, 2, "ffn_W1"),      # m·n = 65536 = N. Splits in 2.
    (512, 128, 1, 2, "ffn_W2"),      # m·n = 65536 = N. Splits in 2.
]

BERT_BASE = [
    (768, 768, 4, 12, "attn_qkvo"),
    (768, 3072, 1, 12, "ffn_W1"),
    (3072, 768, 1, 12, "ffn_W2"),
]


def time_one(be, in_dim, out_dim, num_slots):
    """Time one linear in seconds (median of 3 reps)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3

    # NEXUS path. Need m·n ≤ N/2; if not, must split.
    splits = max(1, (in_dim * out_dim + num_slots - 1) // num_slots)
    out_per = (out_dim + splits - 1) // splits

    W = rng.standard_normal((out_per, in_dim)) * 0.05
    b = rng.standard_normal(out_per) * 0.1

    # warmup + key registration
    ct_x = be.encrypt_coeff(x.tolist())
    _ = be.nexus_linear(ct_x, W, in_dim=in_dim, bias=b.tolist())
    print(f"    [keys={len(be._registered_shifts)}]")
    times = []
    for _ in range(3):
        ct_x = be.encrypt_coeff(x.tolist())
        t = time.time()
        _ = be.nexus_linear(ct_x, W, in_dim=in_dim, bias=b.tolist())
        times.append(time.time() - t)
    return float(np.median(times)), splits


def main():
    print("Init HEonGPU N=2^16 (bootstrap chain)...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    print(f"  num_slots={be._num_slots}")

    print("\n=== BERT-tiny (seq_len=8) ===")
    total = 0.0
    for in_dim, out_dim, count, layers, name in BERT_TINY:
        t1, splits = time_one(be, in_dim, out_dim, be._num_slots)
        per_call = t1 * splits
        # per-token: each token its own ct
        seq = 8
        per_layer = per_call * count * seq
        layer_total = per_layer * layers
        print(f"  {name:14s}  in={in_dim:4d} out={out_dim:4d}  splits={splits}  "
              f"1-call={t1*1000:.0f}ms  per-layer-all-tokens={per_layer:.1f}s  total={layer_total:.1f}s")
        total += layer_total
    print(f"  → linears total ≈ {total:.1f}s  (excluding polynomials, attention QK/V, encoding overhead)")

    # Optional: BERT-base shapes (no e2e, just per-call cost)
    print("\n=== BERT-base shapes (per-call cost only) ===")
    for in_dim, out_dim, count, layers, name in BERT_BASE:
        t1, splits = time_one(be, in_dim, out_dim, be._num_slots)
        per_call = t1 * splits
        seq = 128
        per_layer = per_call * count * seq
        layer_total = per_layer * layers
        print(f"  {name:14s}  in={in_dim:4d} out={out_dim:4d}  splits={splits}  "
              f"1-call={t1*1000:.0f}ms  per-call(all-splits)={per_call*1000:.0f}ms  "
              f"projected-12layer-128tok={layer_total:.0f}s")


if __name__ == "__main__":
    main()
