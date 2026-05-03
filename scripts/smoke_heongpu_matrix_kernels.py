"""GPU smoke test for the 5 new matrix-packed kernels on HEonGPU H100.

Validates correctness against numpy ground truth at small scale (no
bootstrap, fast keygen). Uses N=2^14 — same surface as the token-op
smoke test — so this completes in seconds.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption import ops_matrix as opm


def _absorbed_eval(x, coeffs, interval):
    a, b = interval
    xp = (2.0 * x - (a + b)) / (b - a)
    out = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        out += c * (xp ** i)
    return out


def main() -> int:
    print("Init HEonGPU (N=2^14, sec_none, no bootstrap)...")
    t0 = time.time()
    be = HEonGPUBackend(poly_modulus_degree=1 << 14, sec_none=True)
    print(f"  keygen: {time.time() - t0:.2f}s   slots={be.num_slots}")

    rng = np.random.default_rng(0)
    seq_len, hidden = 4, 16
    block = next_pow2(hidden)

    failed = []

    # 1. enc_gelu_matrix
    print("\n[1] enc_gelu_matrix")
    x = rng.uniform(-2.0, 2.0, (seq_len, hidden))
    coeffs = [0.5, 0.4, 0.0, -0.05]
    interval = (-3.0, 3.0)
    mpt = MatrixPackedTensor.encrypt(be, x, block=block)
    t = time.time()
    y = opm.enc_gelu_matrix(be, mpt, coeffs, interval)
    print(f"  GPU wall: {time.time() - t:.2f}s")
    got = y.decrypt(be)
    ref = _absorbed_eval(x, coeffs, interval)
    err = np.max(np.abs(got - ref))
    print(f"  max-err = {err:.3e}  → {'PASS' if err < 5e-3 else 'FAIL'}")
    if err >= 5e-3:
        failed.append("gelu")

    # 2. enc_softmax_matrix
    print("\n[2] enc_softmax_matrix")
    L = 4
    scores = rng.uniform(-2.0, 2.0, (L, L))
    coeffs_s = [1.0, 0.5, 0.1]
    block_s = next_pow2(L)
    mpt = MatrixPackedTensor.encrypt(be, scores, block=block_s)
    t = time.time()
    y = opm.enc_softmax_matrix(be, mpt, coeffs_s, interval)
    print(f"  GPU wall: {time.time() - t:.2f}s")
    got = y.decrypt(be)
    ref = _absorbed_eval(scores, coeffs_s, interval)
    err = np.max(np.abs(got - ref))
    print(f"  max-err = {err:.3e}  → {'PASS' if err < 5e-3 else 'FAIL'}")
    if err >= 5e-3:
        failed.append("softmax")

    # 3. enc_layernorm_matrix
    print("\n[3] enc_layernorm_matrix")
    xx = rng.standard_normal((seq_len, hidden)) * 0.3
    gamma = rng.uniform(0.5, 1.5, hidden)
    beta = rng.uniform(-0.2, 0.2, hidden)
    vs = np.linspace(0.05, 0.5, 50)
    xpv = (2.0 * vs - 0.55) / 0.45
    Vmat = np.vander(xpv, 4, increasing=True)
    inv_c, *_ = np.linalg.lstsq(Vmat, 1.0 / np.sqrt(vs), rcond=None)
    invsqrt_interval = (0.05, 0.5)
    mpt = MatrixPackedTensor.encrypt(be, xx, block=block)
    t = time.time()
    y = opm.enc_layernorm_matrix(
        be, mpt, inv_c.tolist(), invsqrt_interval, gamma, beta
    )
    print(f"  GPU wall: {time.time() - t:.2f}s")
    got = y.decrypt(be)
    mean = xx.mean(axis=1, keepdims=True)
    centred = xx - mean
    var = (centred ** 2).mean(axis=1, keepdims=True)
    inv_sigma = _absorbed_eval(var, inv_c, invsqrt_interval)
    ref = centred * inv_sigma * gamma + beta
    err = np.max(np.abs(got - ref))
    print(f"  max-err = {err:.3e}  → {'PASS' if err < 5e-2 else 'FAIL'}")
    if err >= 5e-2:
        failed.append("ln")

    # 4. enc_qk_scores_matrix
    print("\n[4] enc_qk_scores_matrix")
    head_dim = 8
    Q = rng.standard_normal((seq_len, head_dim)) * 0.3
    K = rng.standard_normal((seq_len, head_dim)) * 0.3
    bk = next_pow2(head_dim)
    Qm = MatrixPackedTensor.encrypt(be, Q, block=bk)
    Km = MatrixPackedTensor.encrypt(be, K, block=bk)
    scale = 1.0 / np.sqrt(head_dim)
    t = time.time()
    S = opm.enc_qk_scores_matrix(be, Qm, Km, scale)
    print(f"  GPU wall: {time.time() - t:.2f}s")
    got = S.decrypt(be)
    ref = scale * (Q @ K.T)
    err = np.max(np.abs(got - ref))
    print(f"  max-err = {err:.3e}  → {'PASS' if err < 5e-3 else 'FAIL'}")
    if err >= 5e-3:
        failed.append("qk")

    # 5. enc_attention_apply_matrix
    print("\n[5] enc_attention_apply_matrix")
    A = rng.uniform(0.0, 1.0, (seq_len, seq_len))
    A = A / A.sum(axis=1, keepdims=True)
    Vv = rng.standard_normal((seq_len, head_dim)) * 0.3
    ba = next_pow2(seq_len)
    Am = MatrixPackedTensor.encrypt(be, A, block=ba)
    Vm = MatrixPackedTensor.encrypt(be, Vv, block=bk)
    t = time.time()
    out = opm.enc_attention_apply_matrix(be, Am, Vm)
    print(f"  GPU wall: {time.time() - t:.2f}s")
    got = out.decrypt(be)
    ref = A @ Vv
    err = np.max(np.abs(got - ref))
    print(f"  max-err = {err:.3e}  → {'PASS' if err < 5e-3 else 'FAIL'}")
    if err >= 5e-3:
        failed.append("attn")

    print()
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("All matrix kernels PASS on H100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
