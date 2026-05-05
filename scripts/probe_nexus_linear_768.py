"""Probe single nexus_linear at BERT-base attention dimensions.

Goal: measure where time goes for one Wq-shaped matmul (768x768) when
input x is a single 768-vector encoded coeff-style. This is the kernel
we'll batch over 32 inputs.

We need m*n <= N. With m=768, n=768 -> 589824 > N=65536. Must split.
Splits = ceil(589824/65536) = 9. So out_per = ceil(768/9) = 86.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    N = 1 << 16
    log(f"Init HEonGPU N={N} bootstrap chain ...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log("configure_bootstrapping (needed for nexus_linear's CtoS) ...")
    be.configure_bootstrapping()
    log(f"  num_slots={be._num_slots}  N={be._N}")

    in_dim = 768
    out_dim = 768
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3

    # out_per * in_dim must be <= N (strict), so out_per <= N // in_dim
    out_per = N // in_dim
    splits = (out_dim + out_per - 1) // out_per
    log(f"Shape {out_dim}x{in_dim}: m*n={in_dim*out_dim} N={N} -> splits={splits} out_per={out_per}")

    # Encode each split weight once.
    log("Build weight slices ...")
    Ws = []
    for s in range(splits):
        lo = s * out_per
        hi = min(lo + out_per, out_dim)
        Ws.append(rng.standard_normal((hi - lo, in_dim)) * 0.05)

    log("Warmup: register keys via 1 nexus_linear call ...")
    ct_x = be.encrypt_coeff(x.tolist())
    t0 = time.time()
    _ = be.nexus_linear(ct_x, Ws[0], in_dim=in_dim)
    log(f"  warmup: {(time.time()-t0)*1000:.1f}ms  keys={len(be._registered_shifts)}")

    log("\nBaseline single-input full QKV-shaped matmul (all splits):")
    times = []
    for rep in range(2):
        ct_x = be.encrypt_coeff(x.tolist())
        t0 = time.time()
        for s in range(splits):
            _ = be.nexus_linear(ct_x, Ws[s], in_dim=in_dim)
        dt = time.time() - t0
        times.append(dt)
        log(f"  rep{rep}: {dt*1000:.1f}ms ({dt/splits*1000:.1f}ms per split)")
    log(f"\nMedian per-call (one 768x86): {min(times)/splits*1000:.1f}ms")
    log(f"For one 768x768 linear: {min(times)*1000:.1f}ms")
    log(f"For QKVO+W1+W2 per layer (4 + 768x3072 + 3072x768):")
    qkvo = min(times) * 4  # 768x768 each
    # W1: 3072x768 = 12 splits of 256
    # W2: 768x3072 = 36 splits of 21 ... wait no, 768*3072=2359296/65536=36 splits, out_per=22
    log(f"  QKVO: {qkvo*1000:.1f}ms")


if __name__ == "__main__":
    main()
