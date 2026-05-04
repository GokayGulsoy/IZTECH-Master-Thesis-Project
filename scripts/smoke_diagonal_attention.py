"""Phase 7d-1 smoke: diagonal attention parity vs token-loop attention.

Compares enc_self_attention_diagonal against enc_self_attention_matrix
on a tiny BERT-like config (hidden=64, heads=2, seq=8). Reports both
correctness (max-abs error vs torch reference) and wall time.
"""
import time
import numpy as np
import torch

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor
from fhe_thesis.encryption.coefficients import PolyCoeffs
from fhe_thesis.encryption.ops_matrix import (
    enc_self_attention_matrix,
    enc_self_attention_diagonal,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def torch_attn(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, n_heads, sm_coeffs, interval):
    """Single-head→multi-head reference using LPAN softmax poly."""
    L, H = x.shape
    head_dim = H // n_heads
    Q = x @ Wq.T + bq
    K = x @ Wk.T + bk
    V = x @ Wv.T + bv
    out = np.zeros_like(x)
    a, b = interval
    s = 2.0 / (b - a); sh = -(a + b) / (b - a)
    for h in range(n_heads):
        sl, el = h * head_dim, (h + 1) * head_dim
        Qh, Kh, Vh = Q[:, sl:el], K[:, sl:el], V[:, sl:el]
        S = (Qh @ Kh.T) / np.sqrt(head_dim)
        Sn = S * s + sh
        # poly eval
        A = np.zeros_like(S)
        for k, c in enumerate(sm_coeffs):
            A += c * (Sn ** k)
        out[:, sl:el] = A @ Vh
    return out @ Wo.T + bo


def main():
    hidden, n_heads, L = 128, 2, 32
    head_dim = hidden // n_heads
    block = 256
    log(f"Config hidden={hidden} heads={n_heads} L={L} block={block}")

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth}")

    rng = np.random.default_rng(7)
    Wq = rng.standard_normal((hidden, hidden)) * 0.1
    Wk = rng.standard_normal((hidden, hidden)) * 0.1
    Wv = rng.standard_normal((hidden, hidden)) * 0.1
    Wo = rng.standard_normal((hidden, hidden)) * 0.1
    bq = np.zeros(hidden); bk = np.zeros(hidden); bv = np.zeros(hidden); bo = np.zeros(hidden)
    x = rng.standard_normal((L, hidden)) * 0.1

    sm_coeffs = np.array([1.0, 0.5, 0.125])
    sm_interval = (-4.0, 4.0)
    pc = PolyCoeffs(power_coeffs=sm_coeffs, interval=sm_interval, degree=2, per_head=False)

    ref = torch_attn(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, n_heads, sm_coeffs, sm_interval)
    log(f"  ref shape {ref.shape}, range [{ref.min():.3f}, {ref.max():.3f}]")

    ct_x = MatrixPackedTensor.encrypt(be, x, block=block)

    log("Running diagonal attention...")
    t = time.time()
    out_d = enc_self_attention_diagonal(
        be, ct_x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, pc, n_heads,
    )
    t_diag = time.time() - t
    arr_d = out_d.decrypt(be)
    err_d = np.max(np.abs(arr_d - ref))
    log(f"  diagonal: wall={t_diag*1000:.1f}ms  max_err={err_d:.3e}")

    log("Running baseline (token-loop) attention...")
    t = time.time()
    out_b = enc_self_attention_matrix(
        be, ct_x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, pc, n_heads,
    )
    t_base = time.time() - t
    arr_b = out_b.decrypt(be)
    err_b = np.max(np.abs(arr_b - ref))
    log(f"  baseline: wall={t_base*1000:.1f}ms  max_err={err_b:.3e}")

    log("")
    log(f"SPEEDUP: {t_base/t_diag:.2f}x   error parity: diag={err_d:.2e} vs base={err_b:.2e}")
    if err_d > 1e-4:
        log("FAIL — diagonal error too large")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
