"""Phase 7b-BSGS steady-state bench: simulate offline plaintext encoding
(NEXUS pattern), then measure inference-only time.

Realistic deployment: weights are fixed; plaintexts are encoded ONCE at
server startup and cached. Per-inference cost is just the GPU compute
(rotate + mul_plain + add) — which the BSGS recall measurement isolates.
"""
import time
import gc
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.ops_matrix import enc_linear_matrix


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def measure(name, fn):
    t = time.time()
    out = fn()
    dt = time.time() - t
    return out, dt


def main():
    hidden, seq_len = 768, 8

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 14,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready. max_depth={be._max_depth}")
    be.bsgs_diag_cache_enabled = True   # eager pre-encode

    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((4 * hidden, hidden)) * 0.02
    b1 = np.zeros(4 * hidden)
    Wq = rng.standard_normal((hidden, hidden)) * 0.02
    bq = np.zeros(hidden)
    x = rng.standard_normal((seq_len, hidden)) * 0.1

    block = next_pow2(4 * hidden)
    log(f"Encrypt input (block={block})...")
    ct_x = MatrixPackedTensor.encrypt(be, x, block=block)
    d0 = be._ops.depth(ct_x.cts[0])
    log(f"  cts={len(ct_x.cts)}  depth={d0}")

    # ─── Cold pass (encodes diagonals, populates caches) ───
    log("Cold pass: W1 + Wq (encodes diagonals)...")
    t = time.time()
    h1 = enc_linear_matrix(be, ct_x, W1, bias=b1)
    cold_w1 = time.time() - t
    log(f"  cold W1 = {cold_w1:.3f}s  (encoding + compute)")

    t = time.time()
    hq = enc_linear_matrix(be, ct_x, Wq, bias=bq)
    cold_wq = time.time() - t
    log(f"  cold Wq = {cold_wq:.3f}s")

    # ─── Warm pass (cache hit → pure compute floor) ───
    log("Warm pass: same weights, cached plaintexts...")
    t = time.time()
    _ = enc_linear_matrix(be, ct_x, W1, bias=b1)
    warm_w1 = time.time() - t
    log(f"  warm W1 = {warm_w1*1000:.1f} ms")

    t = time.time()
    _ = enc_linear_matrix(be, ct_x, Wq, bias=bq)
    warm_wq = time.time() - t
    log(f"  warm Wq = {warm_wq*1000:.1f} ms")

    # Drop caches before next run.
    be._bsgs_diag_cache.clear()
    gc.collect()

    # ─── Steady-state projection (BERT-base) ───
    n_layers = 12
    # Per layer: 4 attn linears (768x768) + W1 (768x3072) + W2 (3072x768).
    # Attn-linear ≈ Wq cost; W1 ≈ Wq * 4 (3072 cols vs 768 cols + 4x diag count);
    # W2 ≈ Wq * 4 too. Use measured warm_wq and warm_w1 as the floors.
    per_layer_warm = 4 * warm_wq + 2 * warm_w1   # crude: W2 ≈ W1 cost-wise
    total_warm = per_layer_warm * n_layers
    print()
    print("=" * 60)
    print("STEADY-STATE PROJECTION (offline-encoded plaintexts)")
    print("=" * 60)
    print(f"  Warm W1 (768->3072): {warm_w1*1000:.1f} ms")
    print(f"  Warm Wq (768->768):  {warm_wq*1000:.1f} ms")
    print(f"  Per-layer linears (4*Wq + 2*W1): {per_layer_warm*1000:.1f} ms")
    print(f"  12-layer linears:    {total_warm:.2f} s")
    print()
    print(f"  NEXUS published BERT-base e2e: 37 s")
    print(f"  Our linears-only steady state: {total_warm:.2f} s "
          f"({37/max(total_warm,1e-3):.2f}x of NEXUS)")
    print()
    print("  COLD vs WARM ratio (encoding overhead):")
    print(f"    W1: {cold_w1/warm_w1:.0f}x   Wq: {cold_wq/warm_wq:.0f}x")


if __name__ == "__main__":
    main()
