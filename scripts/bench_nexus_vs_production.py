"""Benchmark: NEXUS coeff_matvec_to_slot vs production matmul_plain (BSGS).

Compares the two paths on a representative BERT linear shape.

  - production: matmul_plain (Halevi–Shoup diagonal-encoded BSGS)
                output: slot-encoded ct, slot[i] = (W·x)[i]  for i in [0, m)
  - NEXUS:      coeff_matvec_to_slot (this work)
                output: 2 slot-encoded cts, with (W·x)[i] at slot bit_rev((i+1)·n−1)
                in cts[0] (low half).

Reports wall time and max absolute error vs the plaintext reference.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def bitrev(i: int, bits: int) -> int:
    r = 0
    for b in range(bits):
        r = (r << 1) | ((i >> b) & 1)
    return r


def main():
    rng = np.random.default_rng(0)

    import sys
    in_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    out_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    print(f"shape: in={in_dim}  out={out_dim}  (m·n = {in_dim*out_dim})")

    print("\nInit HEonGPU N=2^16 with bootstrap-ready chain...")
    t = time.time()
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    print(f"  ctx init + keygen: {time.time()-t:.1f}s")
    t = time.time()
    be.configure_bootstrapping()
    print(f"  configure_bootstrapping: {time.time()-t:.1f}s")

    log_n = int(np.log2(be._num_slots))

    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    expected = W @ x

    # Warmup HEonGPU kernels
    print("\n[warmup] one of each path...")
    ct_slot = be.encrypt([float(v) for v in x] + [0.0] * (be._num_slots - in_dim))
    _ = be.matmul_plain(ct_slot, W.tolist())
    ct_coef = be.encrypt_coeff(x.tolist())
    _ = be.coeff_matvec_to_slot(ct_coef, W, in_dim=in_dim)
    print("  done.")

    # ── benchmark production path ──────────────────────────────────
    N_REPS = 5
    print(f"\n[A] production matmul_plain (BSGS, slot-packed)  ({N_REPS} reps)")
    times = []
    for _ in range(N_REPS):
        ct_x = be.encrypt([float(v) for v in x] + [0.0] * (be._num_slots - in_dim))
        t = time.time()
        ct_y = be.matmul_plain(ct_x, W.tolist())
        times.append(time.time() - t)
    times = np.array(times)
    s = np.asarray(be.decrypt(ct_y))[:out_dim]
    err_a = np.max(np.abs(s - expected))
    print(f"  wall:  median={np.median(times)*1000:.2f}ms  min={times.min()*1000:.2f}ms  max={times.max()*1000:.2f}ms")
    print(f"  err vs W·x: {err_a:.3e}")

    # ── benchmark NEXUS coeff_matvec_to_slot ───────────────────────
    print(f"\n[B] NEXUS coeff_matvec_to_slot (depth-0 → CtoS)  ({N_REPS} reps)")
    times = []
    for _ in range(N_REPS):
        ct_x = be.encrypt_coeff(x.tolist())
        t = time.time()
        cts_y = be.coeff_matvec_to_slot(ct_x, W, in_dim=in_dim)
        times.append(time.time() - t)
    times = np.array(times)
    slots = np.asarray(be.decrypt(cts_y[0]))[: be._num_slots]
    target_idx = np.array([bitrev((i + 1) * in_dim - 1, log_n) for i in range(out_dim)])
    s = slots[target_idx]
    err_b = np.max(np.abs(s - expected))
    print(f"  wall:  median={np.median(times)*1000:.2f}ms  min={times.min()*1000:.2f}ms  max={times.max()*1000:.2f}ms")
    print(f"  err vs W·x: {err_b:.3e}")

    # ── summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"shape: ({out_dim}, {in_dim})")
    print(f"  production matmul_plain:       {np.median(times)*0:.2f}ms  err={err_a:.2e}")
    print(f"  NEXUS coeff_matvec_to_slot:    err={err_b:.2e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
