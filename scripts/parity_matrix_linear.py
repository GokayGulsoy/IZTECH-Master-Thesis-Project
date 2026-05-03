"""Parity test for matrix-packed linear vs token-packed linear.

Verifies that ``ops_matrix.enc_linear_matrix`` on a
:class:`MatrixPackedTensor` produces the same numerical output as the
existing ``backend.matmul_plain`` on a :class:`TokenPackedTensor`,
modulo CKKS noise.

Run from the repo root:

    PYTHONPATH=. python scripts/parity_matrix_linear.py            # default 8x32x16
    PYTHONPATH=. python scripts/parity_matrix_linear.py --bert     # 4x768x768
"""

from __future__ import annotations

import argparse
import time
import sys

import numpy as np

from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
from fhe_thesis.encryption.ops_matrix import enc_linear_matrix
from fhe_thesis.encryption.packing import TokenPackedTensor


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bert", action="store_true",
                   help="Use BERT-Base sized layer (slow, ~minutes on CPU).")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--ring", type=int, default=16384)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tol", type=float, default=2e-2)
    args = p.parse_args()

    if args.bert:
        seq_len, in_dim, out_dim = 4, 768, 768
        block = 1024
    else:
        seq_len, in_dim, out_dim = 8, 32, 16
        block = 64

    rng = np.random.default_rng(0)
    X = rng.standard_normal((seq_len, in_dim)).astype(np.float64) * 0.5
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float64) * 0.1
    b = rng.standard_normal((out_dim,)).astype(np.float64) * 0.1

    expected = X @ W.T + b

    num_slots = max(args.ring // 2, block * (seq_len // (args.ring // 2 // block) + 1))
    num_slots = max(num_slots, block)
    # Round up to next power of two to satisfy backend assumptions.
    num_slots = 1 << (num_slots - 1).bit_length()

    print(f"shape: seq={seq_len} in={in_dim} out={out_dim} block={block} "
          f"slots={num_slots} ring={args.ring} depth={args.depth}")
    print("init OpenFHE backend ...")
    t0 = time.time()
    be = OpenFHEBackend(
        multiplicative_depth=args.depth,
        ring_dim=args.ring,
        num_slots=num_slots,
        enable_bootstrap=False,
        num_threads=args.threads,
    )
    print(f"  keygen: {time.time() - t0:.2f}s")

    # ── Reference: TokenPackedTensor matmul ────────────────────────
    print("\n[ref] TokenPackedTensor + backend.matmul_plain ...")
    t0 = time.time()
    tpt = TokenPackedTensor.encrypt(be, X)
    ref_cts = [be.matmul_plain(ct, W.tolist(), b.tolist()) for ct in tpt.cts]
    ref_tpt = TokenPackedTensor.from_ciphertexts(ref_cts, hidden_dim=out_dim)
    ref = ref_tpt.decrypt(be)
    print(f"  wall: {time.time() - t0:.2f}s   shape={ref.shape}")

    # ── Test: MatrixPackedTensor + enc_linear_matrix ───────────────
    print("\n[test] MatrixPackedTensor + enc_linear_matrix ...")
    t0 = time.time()
    mpt = MatrixPackedTensor.encrypt(be, X, block=block)
    print(f"  packing: B={mpt.tokens_per_ct} ciphertexts={len(mpt.cts)}")
    out_mpt = enc_linear_matrix(be, mpt, W, bias=b)
    out = out_mpt.decrypt(be)
    print(f"  wall: {time.time() - t0:.2f}s   shape={out.shape}")

    # ── Compare ──────────────────────────────────────────────────────
    ref_err  = np.max(np.abs(ref - expected))
    test_err = np.max(np.abs(out - expected))
    diff     = np.max(np.abs(ref - out))
    print(f"\nmax|ref  - plaintext|  = {ref_err:.4g}   (existing TokenPackedTensor matmul)")
    print(f"max|test - plaintext|  = {test_err:.4g}   (new matrix-packed enc_linear_matrix)")
    print(f"max|test - ref|        = {diff:.4g}")

    # Success criterion is correctness vs the true plaintext result.
    ok = test_err < args.tol
    print("\nKERNEL CORRECTNESS", "PASS" if ok else "FAIL", f"(tol={args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
