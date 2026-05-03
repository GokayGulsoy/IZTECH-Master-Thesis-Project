"""Parity test for ``enc_linear_matrix`` on the HEonGPU backend.

Mirrors ``scripts/parity_matrix_linear.py`` but uses the GPU backend so
we exercise the full pipeline:

    Python  ->  HEonGPUBackend  ->  pybind11 wrapper  ->  HEonGPU C++
            ->  CUDA kernels    ->  H100 SXM (sm_90)

Run on the Pod from /workspace/repo:

    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \\
        PYTHONPATH=. python scripts/parity_matrix_linear_gpu.py
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \\
        PYTHONPATH=. python scripts/parity_matrix_linear_gpu.py --bert
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor
from fhe_thesis.encryption.ops_matrix import enc_linear_matrix


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bert", action="store_true",
                   help="BERT-Base sized layer (768 -> 768).")
    p.add_argument("--ring", type=int, default=32768)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--prime-bits", type=int, default=40)
    p.add_argument("--tol", type=float, default=5e-2)
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

    print(f"shape: seq={seq_len} in={in_dim} out={out_dim} block={block} "
          f"ring={args.ring} depth={args.depth}")

    print("init HEonGPU backend ...")
    t0 = time.perf_counter()
    Q_bits = [60] + [args.prime_bits] * args.depth
    P_bits = [60]
    be = HEonGPUBackend(
        poly_modulus_degree=args.ring,
        q_prime_bits=Q_bits,
        p_prime_bits=P_bits,
        sec_none=False,
        scale_bits=args.prime_bits,
    )
    print(f"  keygen+galois: {time.perf_counter() - t0:.2f} s "
          f"(slots={be.capabilities.n_slots})")

    print("\nencrypt ...")
    t0 = time.perf_counter()
    mpt = MatrixPackedTensor.encrypt(be, X, block=block)
    print(f"  packing: B={mpt.tokens_per_ct} ciphertexts={len(mpt.cts)}  "
          f"({time.perf_counter() - t0:.2f} s)")

    print("\nenc_linear_matrix on H100 ...")
    t0 = time.perf_counter()
    out_mpt = enc_linear_matrix(be, mpt, W, bias=b)
    dt = time.perf_counter() - t0
    print(f"  wall: {dt:.3f} s   ({dt * 1000 / seq_len:.1f} ms/token)")

    out = out_mpt.decrypt(be)

    err = np.max(np.abs(out - expected))
    rel = err / max(1.0, np.max(np.abs(expected)))
    print(f"\nmax|out - plaintext|  = {err:.4g}   (rel {rel:.4g})")
    ok = err < args.tol
    print("KERNEL CORRECTNESS", "PASS" if ok else "FAIL", f"(tol={args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
