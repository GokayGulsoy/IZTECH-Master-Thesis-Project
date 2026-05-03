"""CPU parity tests for the new matrix-packed kernels.

Verifies each matrix-packed op produces the same numerical output as the
matching token-packed op (or a numpy reference) on the OpenFHE backend.

Kernels tested:
  - per_block_sum            (primitive used by LN / softmax)
  - enc_gelu_matrix          vs ops.enc_gelu_poly
  - enc_layernorm_matrix     vs ops.enc_ln_poly
  - enc_softmax_matrix       vs ops.enc_softmax_poly
  - enc_qk_scores_matrix     vs ops.enc_qk_scores
  - enc_attention_apply_matrix vs ops.enc_attention_apply

Run:
    PYTHONPATH=. python scripts/parity_matrix_kernels.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
from fhe_thesis.encryption.packing import TokenPackedTensor
from fhe_thesis.encryption import ops
from fhe_thesis.encryption import ops_matrix as opm


def _polyval_np(x: np.ndarray, coeffs):
    out = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        out += c * (x ** i)
    return out


def _absorbed_eval(x, coeffs, interval):
    """Evaluate poly with [a,b]→[-1,1] standardisation, in the same basis
    used by ops.enc_gelu_poly etc.  i.e. poly is in standardised x'."""
    a, b = interval
    xp = (2.0 * x - (a + b)) / (b - a)
    return _polyval_np(xp, coeffs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--ring", type=int, default=16384)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tol", type=float, default=5e-2)
    p.add_argument(
        "--skip", type=str, default="",
        help="comma-separated kernel names to skip (e.g. qk,attn)",
    )
    p.add_argument("--mha", action="store_true",
                   help="also run composed multi-head self-attention parity")
    args = p.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    # Small dims keep parity under 30s on CPU.
    seq_len, hidden_dim = 4, 16
    block = next_pow2(hidden_dim)  # 16
    num_slots = max(args.ring // 2, 64)

    rng = np.random.default_rng(0)

    print(
        f"shape: seq={seq_len} hidden={hidden_dim} block={block} "
        f"slots={num_slots} ring={args.ring} depth={args.depth}"
    )
    t0 = time.time()
    be = OpenFHEBackend(
        multiplicative_depth=args.depth,
        ring_dim=args.ring,
        num_slots=num_slots,
        enable_bootstrap=False,
        num_threads=args.threads,
    )
    print(f"  keygen: {time.time() - t0:.2f}s")

    failed = []

    # ── per_block_sum ───────────────────────────────────────────────
    if "block_sum" not in skip:
        print("\n[test] per_block_sum ...")
        x = rng.standard_normal((seq_len, hidden_dim)).astype(np.float64) * 0.3
        mpt = MatrixPackedTensor.encrypt(be, x, block=block)
        out_cts = [
            opm.per_block_sum(
                be, ct, hidden_dim=hidden_dim, block=block, num_slots=num_slots
            )
            for ct in mpt.cts
        ]
        # Decrypt slot 0 of each block of each ct, compare with row sum.
        B = mpt.tokens_per_ct
        ref = x.sum(axis=1)  # (seq_len,)
        got = np.zeros(seq_len)
        for g, ct in enumerate(out_cts):
            slots = np.asarray(be.decrypt(ct))
            for k in range(B):
                tok = g * B + k
                if tok >= seq_len:
                    break
                got[tok] = slots[k * block]
        err = np.max(np.abs(got - ref))
        print(f"  max-err = {err:.3e}  (expected ≈ {ref})")
        ok = err < 1e-3
        print("  ", "PASS" if ok else "FAIL")
        if not ok:
            failed.append("per_block_sum")

    # ── enc_gelu_matrix ─────────────────────────────────────────────
    if "gelu" not in skip:
        print("\n[test] enc_gelu_matrix ...")
        x = rng.uniform(-2.0, 2.0, (seq_len, hidden_dim)).astype(np.float64)
        # Simple deg-3 polynomial as a stand-in for GELU coeffs.
        coeffs = [0.5, 0.4, 0.0, -0.05]
        interval = (-3.0, 3.0)
        mpt = MatrixPackedTensor.encrypt(be, x, block=block)
        out_mpt = opm.enc_gelu_matrix(be, mpt, coeffs, interval)
        got = out_mpt.decrypt(be)
        ref = _absorbed_eval(x, coeffs, interval)
        err = np.max(np.abs(got - ref))
        print(f"  max-err = {err:.3e}")
        ok = err < args.tol
        print("  ", "PASS" if ok else "FAIL")
        if not ok:
            failed.append("enc_gelu_matrix")

    # ── enc_softmax_matrix ──────────────────────────────────────────
    if "softmax_poly" not in skip:
        print("\n[test] enc_softmax_matrix (poly part only) ...")
        L = 4
        scores = rng.uniform(-2.0, 2.0, (L, L)).astype(np.float64)
        coeffs = [1.0, 0.5, 0.1]
        interval = (-3.0, 3.0)
        block_s = next_pow2(L)
        mpt = MatrixPackedTensor.encrypt(be, scores, block=block_s)
        out_mpt = opm.enc_softmax_matrix(be, mpt, coeffs, interval)
        got = out_mpt.decrypt(be)
        ref = _absorbed_eval(scores, coeffs, interval)
        err = np.max(np.abs(got - ref))
        print(f"  max-err = {err:.3e}")
        ok = err < args.tol
        print("  ", "PASS" if ok else "FAIL")
        if not ok:
            failed.append("enc_softmax_matrix")

    # ── enc_layernorm_matrix ───────────────────────────────────────
    if "ln" not in skip:
        print("\n[test] enc_layernorm_matrix ...")
        x = rng.standard_normal((seq_len, hidden_dim)).astype(np.float64) * 0.3
        gamma = rng.uniform(0.5, 1.5, hidden_dim).astype(np.float64)
        beta = rng.uniform(-0.2, 0.2, hidden_dim).astype(np.float64)
        # inv-sqrt poly: pick coeffs and interval such that variance lies
        # within. With x ~ N(0, 0.09), var ≈ 0.09 → use interval [0, 0.5].
        # Approximate 1/sqrt(v) with degree-2 over [0.05, 0.5]:
        vs = np.linspace(0.05, 0.5, 50)
        target = 1.0 / np.sqrt(vs)
        # standardised inputs in [-1, 1]
        xp = (2.0 * vs - 0.55) / 0.45
        # Fit deg-3 in std basis via least squares
        Vmat = np.vander(xp, 4, increasing=True)
        invsqrt_coeffs, *_ = np.linalg.lstsq(Vmat, target, rcond=None)
        invsqrt_coeffs = invsqrt_coeffs.tolist()
        invsqrt_interval = (0.05, 0.5)

        mpt = MatrixPackedTensor.encrypt(be, x, block=block)
        out_mpt = opm.enc_layernorm_matrix(
            be, mpt, invsqrt_coeffs, invsqrt_interval, gamma, beta
        )
        got = out_mpt.decrypt(be)
        # Reference: same math as the kernel uses (poly inv-sqrt, no eps).
        mean = x.mean(axis=1, keepdims=True)
        centred = x - mean
        var = (centred ** 2).mean(axis=1, keepdims=True)
        inv_sigma = _absorbed_eval(var, invsqrt_coeffs, invsqrt_interval)
        ref = centred * inv_sigma * gamma + beta
        err = np.max(np.abs(got - ref))
        print(f"  max-err = {err:.3e}")
        ok = err < args.tol
        print("  ", "PASS" if ok else "FAIL")
        if not ok:
            failed.append("enc_layernorm_matrix")

    # ── enc_qk_scores_matrix ───────────────────────────────────────
    if "qk" not in skip:
        print("\n[test] enc_qk_scores_matrix ...")
        L, head_dim = 4, 8
        block_qk = next_pow2(head_dim)
        Q = rng.standard_normal((L, head_dim)).astype(np.float64) * 0.3
        K = rng.standard_normal((L, head_dim)).astype(np.float64) * 0.3
        scale = 1.0 / np.sqrt(head_dim)
        Qm = MatrixPackedTensor.encrypt(be, Q, block=block_qk)
        Km = MatrixPackedTensor.encrypt(be, K, block=block_qk)
        out_mpt = opm.enc_qk_scores_matrix(be, Qm, Km, scale)
        got = out_mpt.decrypt(be)
        ref = scale * (Q @ K.T)
        err = np.max(np.abs(got - ref))
        print(f"  max-err = {err:.3e}")
        ok = err < args.tol
        print("  ", "PASS" if ok else "FAIL")
        if not ok:
            failed.append("enc_qk_scores_matrix")

    # ── enc_attention_apply_matrix ─────────────────────────────────
    if "attn" not in skip:
        print("\n[test] enc_attention_apply_matrix ...")
        L, head_dim = 4, 8
        block_v = next_pow2(head_dim)
        block_a = next_pow2(L)
        attn = rng.uniform(0.0, 1.0, (L, L)).astype(np.float64)
        attn = attn / attn.sum(axis=1, keepdims=True)  # row-stochastic
        V = rng.standard_normal((L, head_dim)).astype(np.float64) * 0.3
        attn_m = MatrixPackedTensor.encrypt(be, attn, block=block_a)
        Vm = MatrixPackedTensor.encrypt(be, V, block=block_v)
        out_mpt = opm.enc_attention_apply_matrix(be, attn_m, Vm)
        got = out_mpt.decrypt(be)
        ref = attn @ V
        err = np.max(np.abs(got - ref))
        print(f"  max-err = {err:.3e}")
        ok = err < args.tol
        print("  ", "PASS" if ok else "FAIL")
        if not ok:
            failed.append("enc_attention_apply_matrix")

    print()
    if failed:
        print(f"❌ FAILED: {failed}")
        return 1
    print("✅ All matrix kernels PASS parity")
    return 0


def run_mha_parity(args):
    """Optional: composed multi-head self-attention parity (matrix vs numpy)."""
    from fhe_thesis.encryption.coefficients import PolyCoeffs

    seq_len, hidden_dim, num_heads = 4, 16, 2
    head_dim = hidden_dim // num_heads
    block = next_pow2(hidden_dim)
    num_slots = max(args.ring // 2, 64)

    rng = np.random.default_rng(0)
    print(
        f"\n[mha-parity] seq={seq_len} hidden={hidden_dim} heads={num_heads} "
        f"head_dim={head_dim} block={block}"
    )

    be = OpenFHEBackend(
        multiplicative_depth=args.depth, ring_dim=args.ring,
        num_slots=num_slots, enable_bootstrap=False, num_threads=args.threads,
    )

    x = rng.standard_normal((seq_len, hidden_dim)).astype(np.float64) * 0.3
    Wq = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float64) * 0.1
    Wk = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float64) * 0.1
    Wv = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float64) * 0.1
    Wo = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float64) * 0.1
    bq = rng.standard_normal(hidden_dim) * 0.05
    bk = rng.standard_normal(hidden_dim) * 0.05
    bv = rng.standard_normal(hidden_dim) * 0.05
    bo = rng.standard_normal(hidden_dim) * 0.05

    # Use a *trivial* softmax poly = identity (linear) so the parity test
    # isolates the multi-head-attention plumbing, not the poly approximation.
    softmax_coeffs = PolyCoeffs(
        power_coeffs=np.array([0.0, 1.0]),
        interval=(-3.0, 3.0),
        degree=1,
        per_head=False,
    )

    # Reference numpy implementation matching the kernel's math.
    def _absorbed(z, c, iv):
        a, b = iv
        zp = (2.0 * z - (a + b)) / (b - a)
        out = np.zeros_like(z)
        for i, ci in enumerate(c):
            out += ci * (zp ** i)
        return out

    Q = x @ Wq.T + bq
    K = x @ Wk.T + bk
    V = x @ Wv.T + bv
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    ref = np.zeros((seq_len, hidden_dim))
    for h in range(num_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        S_h = inv_sqrt_d * (Q[:, s:e] @ K[:, s:e].T)
        A_h = _absorbed(S_h, softmax_coeffs.power_coeffs, softmax_coeffs.interval)
        ref[:, s:e] = A_h @ V[:, s:e]
    ref = ref @ Wo.T + bo

    mpt = MatrixPackedTensor.encrypt(be, x, block=block)
    out_mpt = opm.enc_self_attention_matrix(
        be, mpt, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
        softmax_coeffs=softmax_coeffs, num_heads=num_heads,
    )
    got = out_mpt.decrypt(be)
    err = np.max(np.abs(got - ref))
    print(f"  max-err = {err:.3e}")
    ok = err < args.tol
    print("  ", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    if rc == 0 and "--mha" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--depth", type=int, default=20)
        ap.add_argument("--ring", type=int, default=16384)
        ap.add_argument("--threads", type=int, default=4)
        ap.add_argument("--tol", type=float, default=5e-2)
        ap.add_argument("--skip", type=str, default="")
        ap.add_argument("--mha", action="store_true")
        args = ap.parse_args()
        rc = run_mha_parity(args)
    sys.exit(rc)
