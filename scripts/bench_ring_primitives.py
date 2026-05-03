"""Phase 3 micro-benchmark: ring_dim 65536 vs 32768 primitive-op latency.

Run locally to estimate the wall-clock impact of --fast-ring before
launching a full BERT-FHE benchmark on the pod.

Usage:
    PYTHONPATH=. python scripts/bench_ring_primitives.py
    PYTHONPATH=. python scripts/bench_ring_primitives.py --threads 32
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def bench(label: str, ring_dim: int, mult_depth: int, num_slots: int,
          threads: int, ops_iters: int = 50):
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend

    print(f"\n========== {label}  N={ring_dim}  depth={mult_depth} ==========")
    t0 = time.time()
    be = OpenFHEBackend(
        multiplicative_depth=mult_depth, ring_dim=ring_dim,
        num_slots=num_slots, enable_bootstrap=False, num_threads=threads,
    )
    t_keygen = time.time() - t0
    print(f"keygen:       {t_keygen:6.2f}s")

    v = np.random.randn(num_slots).astype(np.float64)
    pt = np.random.randn(num_slots).astype(np.float64)

    t0 = time.time()
    for _ in range(20):
        ct0 = be.encrypt(v)
    print(f"encrypt:      {(time.time() - t0) / 20 * 1000:6.2f}ms/op")

    t0 = time.time()
    for _ in range(20):
        _ = be.decrypt(ct0)
    print(f"decrypt:      {(time.time() - t0) / 20 * 1000:6.2f}ms/op")

    t0 = time.time()
    for _ in range(ops_iters):
        ct1 = be.mul_plain(ct0, pt)
    print(f"mul_plain:    {(time.time() - t0) / ops_iters * 1000:6.2f}ms/op")

    ct_a = be.encrypt(v)
    ct_b = be.encrypt(np.random.randn(num_slots).astype(np.float64))
    t0 = time.time()
    for _ in range(ops_iters):
        _ = be.mul(ct_a, ct_b)
    print(f"mul (ct*ct):  {(time.time() - t0) / ops_iters * 1000:6.2f}ms/op  (incl. relin)")

    if hasattr(be, "rotate"):
        t0 = time.time()
        for _ in range(ops_iters):
            _ = be.rotate(ct0, 1)
        print(f"rotate(1):    {(time.time() - t0) / ops_iters * 1000:6.2f}ms/op")

    out = np.array(be.decrypt(ct1))[:num_slots]
    expect = v * pt
    rel_err = np.linalg.norm(out - expect) / (np.linalg.norm(expect) + 1e-12)
    print(f"numerical:    rel_err = {rel_err:.2e}  (mul_plain decrypt)")
    return t_keygen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num-slots", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    t_std = bench("STD     (current LPAN config)",
                  ring_dim=1 << 16, mult_depth=25,
                  num_slots=args.num_slots, threads=args.threads, ops_iters=args.iters)
    t_red = bench("REDUCED (Phase 3 fast preset)",
                  ring_dim=1 << 15, mult_depth=18,
                  num_slots=args.num_slots, threads=args.threads, ops_iters=args.iters)
    print(f"\n=== keygen speedup: {t_std / t_red:.2f}x ===")


if __name__ == "__main__":
    main()
