"""Validate nexus_linear: coefficient input → contiguous slot-encoded output.

End-to-end check that nexus_linear(W, x) gives slot[i] = (W·x)[i] for i ∈ [0, m).
Also benchmarks against production matmul_plain.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    rng = np.random.default_rng(0)
    in_dim, out_dim = 128, 128

    print("Init HEonGPU N=2^16 with bootstrap-ready chain...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()

    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    b = rng.standard_normal(out_dim) * 0.1
    expected = W @ x + b

    print(f"\nshape: in={in_dim}  out={out_dim}")
    print(f"  expected[:4] = {expected[:4]}")

    # Warmup
    ct_x_coef = be.encrypt_coeff(x.tolist())
    _ = be.nexus_linear(ct_x_coef, W, in_dim=in_dim, bias=b.tolist())

    # ── nexus_linear ──────────────────────────────────────────────
    print("\n[A] NEXUS linear (coeff_matvec + CtoS + gather)")
    times = []
    for _ in range(3):
        ct_x_coef = be.encrypt_coeff(x.tolist())
        t = time.time()
        ct_out = be.nexus_linear(ct_x_coef, W, in_dim=in_dim, bias=b.tolist())
        times.append(time.time() - t)
    times = np.array(times)
    s = np.asarray(be.decrypt(ct_out))[:out_dim]
    err = np.max(np.abs(s - expected))
    print(f"  wall: median={np.median(times)*1000:.1f}ms  min={times.min()*1000:.1f}ms  max={times.max()*1000:.1f}ms")
    print(f"  err vs (W·x + b): {err:.3e}")
    print(f"  got[:4] = {s[:4]}")

    # ── production matmul_plain for reference ─────────────────────
    print("\n[B] Production matmul_plain (BSGS, slot-packed)")
    ct_x_slot = be.encrypt([float(v) for v in x] + [0.0] * (be._num_slots - in_dim))
    _ = be.matmul_plain(ct_x_slot, W.tolist(), bias=b.tolist())  # warmup
    times = []
    for _ in range(3):
        ct_x_slot = be.encrypt([float(v) for v in x] + [0.0] * (be._num_slots - in_dim))
        t = time.time()
        ct_out = be.matmul_plain(ct_x_slot, W.tolist(), bias=b.tolist())
        times.append(time.time() - t)
    times = np.array(times)
    s = np.asarray(be.decrypt(ct_out))[:out_dim]
    err = np.max(np.abs(s - expected))
    print(f"  wall: median={np.median(times)*1000:.1f}ms  min={times.min()*1000:.1f}ms  max={times.max()*1000:.1f}ms")
    print(f"  err vs (W·x + b): {err:.3e}")


if __name__ == "__main__":
    main()
