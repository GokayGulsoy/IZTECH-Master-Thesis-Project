"""Measure overhead of converting between slot ↔ coeff encoding.

If slot_to_coeff (input adapter for NEXUS) is too expensive, NEXUS
linears can't be drop-in replacements for the existing slot-based
pipeline.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(be._num_slots) * 0.3
    ct_slot = be.encrypt(x.tolist())  # slot-encoded
    print(f"  slot ct depth={be._ops.depth(ct_slot)}")

    # We need 2 inputs for StoC (real + imag halves). Use same ct twice
    # for benchmarking purposes.
    ct0 = be._ops.clone_ct(ct_slot)
    ct1 = be._ops.clone_ct(ct_slot)

    # Drop to StoC level.
    target_lvl = be._ops.slot_to_coeff_level()
    print(f"  StoC requires depth ≤ {30 - target_lvl}")
    # Drop to that depth.
    while be._ops.depth(ct0) < (30 - target_lvl):
        be._ops.mod_drop_inplace_ct(ct0)
        be._ops.mod_drop_inplace_ct(ct1)

    # warmup
    _ = be._ops.slot_to_coeff(ct0, ct1, be._gk)
    times = []
    for _ in range(3):
        ct0 = be._ops.clone_ct(ct_slot)
        ct1 = be._ops.clone_ct(ct_slot)
        while be._ops.depth(ct0) < (30 - target_lvl):
            be._ops.mod_drop_inplace_ct(ct0)
            be._ops.mod_drop_inplace_ct(ct1)
        t = time.time()
        _ = be._ops.slot_to_coeff(ct0, ct1, be._gk)
        times.append(time.time() - t)
    print(f"  slot_to_coeff: median={np.median(times)*1000:.1f}ms")

    # Compare to CtoS.
    ct0 = be.encrypt_coeff(x.tolist())
    _ = be._ops.coeff_to_slot(ct0, be._gk)
    times = []
    for _ in range(3):
        ct0 = be.encrypt_coeff(x.tolist())
        t = time.time()
        _ = be._ops.coeff_to_slot(ct0, be._gk)
        times.append(time.time() - t)
    print(f"  coeff_to_slot: median={np.median(times)*1000:.1f}ms")


if __name__ == "__main__":
    main()
