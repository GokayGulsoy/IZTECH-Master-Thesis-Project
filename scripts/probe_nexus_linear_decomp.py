"""Decompose nexus_linear cost: weight encode vs per-input compute.

This determines amortization potential. If encode dominates and the
per-input ct·pt+CtoS+gather is fast, batching B inputs across one
W-encoding gives ~B× speedup.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    N = 1 << 16
    log(f"Init N={N}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()

    in_dim = 768
    out_per = N // in_dim   # 85
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_per, in_dim)) * 0.05

    # Build the W-poly (matches coeff_matvec internal layout).
    def build_w_poly(W, in_dim, N):
        m, n = W.shape
        wp = [0.0] * N
        for i in range(m):
            base = i * n
            for j in range(n):
                wp[base + (n - 1 - j)] = float(W[i, j])
        return wp

    log("Warmup nexus_linear (registers Galois keys + caches)...")
    ct_x = be.encrypt_coeff(x.tolist())
    _ = be.nexus_linear(ct_x, W, in_dim=in_dim)
    log(f"  keys={len(be._registered_shifts)}")

    # Decompose: stage 1 = build w_poly (Python), stage 2 = encode_coeff (GPU NTT)
    # stage 3 = mod-drop pt + multiply_plain + clear rescale + CtoS + rescales
    # stage 4 = gather_slots
    log("\n--- Decomposition (median of 3) ---")

    # Stage A: build w_poly (Python)
    times_a = []
    for _ in range(3):
        t = time.time()
        wp = build_w_poly(W, in_dim, N)
        times_a.append(time.time()-t)
    log(f"  A. build w_poly (Python list): {min(times_a)*1000:.1f}ms")

    # Stage B: encode_coeff (GPU)
    times_b = []
    for _ in range(3):
        t = time.time()
        pt_w = be._encode_coeff_pad(wp)
        times_b.append(time.time()-t)
    log(f"  B. encode_coeff_pad (GPU NTT): {min(times_b)*1000:.1f}ms")

    # Stage C: clone + multiply_plain + rescale-clear + CtoS + 2 rescales
    times_c = []
    for _ in range(3):
        ct_x = be.encrypt_coeff(x.tolist())
        pt_w = be._encode_coeff_pad(wp)
        t = time.time()
        out = be._ops.clone_ct(ct_x)
        be._ops.multiply_plain_inplace(out, pt_w)
        be._ops.clear_rescale_required(out)
        cts = be.coeff_to_slot(out)
        for c in cts:
            be._ops.set_rescale_required(c)
            be._ops.rescale_inplace(c)
        times_c.append(time.time()-t)
    log(f"  C. mul + CtoS + rescales:       {min(times_c)*1000:.1f}ms")

    # Stage D: gather (read-from-bitrev)
    targets = be.nexus_target_slots(in_dim, out_per)
    times_d = []
    for _ in range(3):
        ct_x = be.encrypt_coeff(x.tolist())
        cts = be.coeff_matvec_to_slot(ct_x, W, in_dim=in_dim)
        t = time.time()
        _ = be.gather_slots(cts[0], targets)
        times_d.append(time.time()-t)
    log(f"  D. gather_slots (BSGS):         {min(times_d)*1000:.1f}ms")

    total_per_input = min(times_b) + min(times_c) + min(times_d)
    amortizable = min(times_b)  # only encoding is reusable across inputs
    per_input = min(times_c) + min(times_d)
    log(f"\n  Total single-input:           {total_per_input*1000:.1f}ms")
    log(f"  Amortizable (W encode):       {amortizable*1000:.1f}ms")
    log(f"  Non-amortizable (per input):  {per_input*1000:.1f}ms")
    log(f"  Asymptotic batch-32 amortized: {(amortizable/32 + per_input)*1000:.1f}ms")
    log(f"  Speedup at B=32:              {total_per_input / (amortizable/32 + per_input):.2f}x")


if __name__ == "__main__":
    main()
