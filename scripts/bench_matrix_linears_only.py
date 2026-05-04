"""Phase 7a: measure matrix-packed linear-only stages on real BERT-base shapes.

Skips LN/GELU/attention. Just W1 then W2 (the FFN linears) on a fresh ct.
Goal: see how many chain levels each stage consumes — the budget needed
between bootstraps.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.ops_matrix import enc_linear_matrix


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    hidden, seq_len = 768, 8

    log("Init backend (no bootstrap config — just measure depth)...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 14,  # ~12 usable levels (BSGS + replicate)
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready. max_depth={be._max_depth}")

    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((4 * hidden, hidden)) * 0.02
    b1 = np.zeros(4 * hidden)
    W2 = rng.standard_normal((hidden, 4 * hidden)) * 0.02
    b2 = np.zeros(hidden)
    x = rng.standard_normal((seq_len, hidden)) * 0.1

    block = next_pow2(4 * hidden)  # 4096
    log(f"Encrypt input (block={block}, seq={seq_len}, hidden={hidden})...")
    ct_x = MatrixPackedTensor.encrypt(be, x, block=block)
    log(f"  cts={len(ct_x.cts)}  B={ct_x.tokens_per_ct}")
    d0 = be._ops.depth(ct_x.cts[0])
    log(f"  initial depth: {d0}")

    log("W1 (hidden=768 → 3072) ...")
    t = time.time()
    h = enc_linear_matrix(be, ct_x, W1, bias=b1)
    dt1 = time.time() - t
    d1 = be._ops.depth(h.cts[0])
    log(f"  W1 wall={dt1:.3f}s  depth={d1}  (Δ={d1-d0})")

    log("W2 (hidden=3072 → 768) ...")
    t = time.time()
    h2 = enc_linear_matrix(be, h, W2, bias=b2)
    dt2 = time.time() - t
    d2 = be._ops.depth(h2.cts[0])
    log(f"  W2 wall={dt2:.3f}s  depth={d2}  (Δ={d2-d1})")

    # All 4 attention linears at hidden×hidden too
    Wq = rng.standard_normal((hidden, hidden)) * 0.02
    bq = np.zeros(hidden)
    log("Wq (hidden=768 → 768) ...")
    t = time.time()
    hq = enc_linear_matrix(be, ct_x, Wq, bias=bq)
    dtq = time.time() - t
    dq = be._ops.depth(hq.cts[0])
    log(f"  Wq wall={dtq:.3f}s  depth={dq}  (Δ={dq-d0})")

    n_layers = 12
    print()
    print("=== Summary (per-layer FFN linears, matrix-packed) ===")
    print(f"  W1: {dt1*1000:.1f}ms  Δdepth={d1-d0}")
    print(f"  W2: {dt2*1000:.1f}ms  Δdepth={d2-d1}")
    print(f"  Wq (1 of 4 attn linears): {dtq*1000:.1f}ms  Δdepth={dq-d0}")
    print()
    print("=== Projection (linears only, no LN/GELU/attn loop, no bootstrap overhead) ===")
    per_layer_lin = (dt1 + dt2 + 4*dtq)
    full = per_layer_lin * n_layers
    print(f"  Per-layer linear total: {per_layer_lin:.3f}s")
    print(f"  12-layer linear total:  {full:.3f}s = {full/60:.1f} min")
    print()
    print(f"  NEXUS published BERT-base e2e: 37s")
    print(f"  Our linears-only headroom: {37/full:.2f}x of NEXUS total wall")


if __name__ == "__main__":
    main()
