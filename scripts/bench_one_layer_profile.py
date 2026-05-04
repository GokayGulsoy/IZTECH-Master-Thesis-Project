"""Profile per-stage cost of one BERT-base layer (matrix-packed) so we
can prioritise Phase 7d.

Measures: 4× attention linears, attention scores, softmax poly, attention
apply, output linear, residual+LN, FFN W1, GELU, FFN W2, residual+LN.

Uses Phase 7b (BSGS + caches) and Phase 7c (low-depth LN). Diag cache
disabled (cold timings) to reflect single-pass inference.
"""
import time, gc
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.ops_matrix import (
    enc_linear_matrix, enc_layernorm_matrix, enc_gelu_matrix,
    enc_softmax_matrix, enc_qk_scores_matrix, enc_attention_apply_matrix,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # BERT-base small-test config: hidden=128, 2 heads, head_dim=64,
    # seq=8 — keeps attention loops manageable for a profiling pass.
    hidden, n_heads, seq_len = 128, 2, 8
    head_dim = hidden // n_heads
    block = 1024
    log(f"Config: hidden={hidden}, n_heads={n_heads}, head_dim={head_dim}, "
        f"seq={seq_len}, block={block}")

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 40,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready. max_depth={be._max_depth}")

    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((hidden, hidden)) * 0.05
    Wk = rng.standard_normal((hidden, hidden)) * 0.05
    Wv = rng.standard_normal((hidden, hidden)) * 0.05
    Wo = rng.standard_normal((hidden, hidden)) * 0.05
    W1 = rng.standard_normal((4 * hidden, hidden)) * 0.05
    W2 = rng.standard_normal((hidden, 4 * hidden)) * 0.05
    bz = lambda d: np.zeros(d)
    gamma = np.ones(hidden); beta = np.zeros(hidden)
    x = rng.standard_normal((seq_len, hidden)) * 0.1

    log("Encrypt input...")
    ct_x = MatrixPackedTensor.encrypt(be, x, block=block)
    times = {}

    def stage(name, fn):
        gc.collect()
        t = time.time()
        out = fn()
        dt = time.time() - t
        times[name] = dt
        d = be._ops.depth(out.cts[0]) if hasattr(out, "cts") else "-"
        log(f"  {name:<22s} {dt*1000:8.1f} ms   depth={d}")
        return out

    log("=== ATTENTION ===")
    Q = stage("Wq",      lambda: enc_linear_matrix(be, ct_x, Wq, bias=bz(hidden)))
    K = stage("Wk",      lambda: enc_linear_matrix(be, ct_x, Wk, bias=bz(hidden)))
    V = stage("Wv",      lambda: enc_linear_matrix(be, ct_x, Wv, bias=bz(hidden)))

    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    S = stage("qk_scores", lambda: enc_qk_scores_matrix(be, Q, K, scale=inv_sqrt_d))

    # Tiny softmax poly (we just want timing, not accuracy here).
    sm_coeffs = [1.0, 0.5, 0.125]
    Asm = stage("softmax_poly", lambda: enc_softmax_matrix(be, S, sm_coeffs, interval=(-4.0, 4.0)))

    AV = stage("attn_apply", lambda: enc_attention_apply_matrix(be, Asm, V))
    O = stage("Wo",      lambda: enc_linear_matrix(be, AV, Wo, bias=bz(hidden)))

    log("=== LN1 (low-depth) ===")
    LN1 = stage("layernorm_1", lambda: enc_layernorm_matrix(
        be, O, gamma=gamma, beta=beta,
        invsqrt_power_coeffs=[1.0, -0.5, 0.375], invsqrt_interval=(0.01, 4.0)))

    log("=== FFN ===")
    H1 = stage("W1", lambda: enc_linear_matrix(be, LN1, W1, bias=bz(4*hidden)))
    gelu_coeffs = [0.5, 0.398, 0.0, -0.0066]
    G = stage("gelu", lambda: enc_gelu_matrix(be, H1, gelu_coeffs, interval=(-4.0, 4.0)))
    H2 = stage("W2", lambda: enc_linear_matrix(be, G, W2, bias=bz(hidden)))

    log("=== LN2 ===")
    LN2 = stage("layernorm_2", lambda: enc_layernorm_matrix(
        be, H2, gamma=gamma, beta=beta,
        invsqrt_power_coeffs=[1.0, -0.5, 0.375], invsqrt_interval=(0.01, 4.0)))

    print()
    print("=" * 60)
    print(f"PER-LAYER PROFILE (hidden={hidden}, seq={seq_len})")
    print("=" * 60)
    total = sum(times.values())
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / total
        bar = "#" * int(pct // 2)
        print(f"  {k:<22s} {v*1000:7.1f} ms  {pct:5.1f}%  {bar}")
    print(f"  {'TOTAL':<22s} {total*1000:7.1f} ms")
    print()
    print(f"Projected 12-layer wall: {total*12:.1f} s  (NEXUS BERT-base e2e: 37s)")


if __name__ == "__main__":
    main()
