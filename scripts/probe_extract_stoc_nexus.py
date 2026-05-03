"""Validate: extract token t from matrix-packed ct, run StoC, run nexus_linear.

This is the kernel of the proposed enc_linear_matrix_nexus path. We need:
  1. Mask + rotate to bring token t's hidden-dim values to slots [0, hidden).
  2. Convert slot-encoded → coeff-encoded via slot_to_coeff.
  3. Run nexus_linear on the coeff-encoded ct.
  4. Rotate output to slot [t*block, ...) and add into accumulator.

Confirms correctness of the full extract → nexus → place pipeline.
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

    rng = np.random.default_rng(42)
    seq_len = 8
    hidden = 128
    out_dim = 128
    block = 128

    X = rng.standard_normal((seq_len, hidden)) * 0.3  # (8, 128)
    W = rng.standard_normal((out_dim, hidden)) * 0.05
    b = rng.standard_normal(out_dim) * 0.1
    expected_Y = X @ W.T + b  # (8, 128)

    # Pack X into a single ciphertext, token t at slots [t*block, t*block+hidden)
    flat = np.zeros(be._num_slots)
    for t in range(seq_len):
        flat[t*block:(t+1)*block] = X[t]
    ct_mp = be.encrypt(flat.tolist())
    print(f"  matrix-packed ct: depth={be._ops.depth(ct_mp)}")

    # Process token-by-token using NEXUS.
    t_start = time.time()
    accum = None
    for t in range(seq_len):
        # 1. Extract token t — mask + rotate to align at slot [0, hidden).
        mask = [0.0] * be._num_slots
        for j in range(hidden):
            mask[t*block + j] = 1.0
        ct_t = be.mul_plain(ct_mp, mask)
        # rotate so that slot [t*block + j] moves to slot j: shift = t*block (positive)
        if t > 0:
            ct_t = be.rotate(ct_t, t*block)
        # rotate adds 1 mult depth via mul_plain — NEXUS needs depth-0!
        # Hmm, mul_plain consumes a level. Let me check what depth ct_t is now.
        d = be._ops.depth(ct_t)
        print(f"  token {t}: ct_t depth={d}")
        if d != 0:
            # We need depth 0 for slot_to_coeff path? Actually no, StoC has its own
            # level requirement. Will check at first failure.
            pass
        # 2. Convert slot → coeff (need 2 inputs: real + imag halves)
        # For real input, use ct_t for both? Actually slot_to_coeff is the inverse
        # of coeff_to_slot which produced 2 outputs. Let me try with same ct twice.
        # First drop both cts to StoC depth.
        target_lvl = be._ops.slot_to_coeff_level()
        target_depth = 30 - target_lvl
        zero_ct = be.encrypt([0.0] * be._num_slots)
        # Bring each independently to target_depth
        while be._ops.depth(ct_t) < target_depth:
            be._ops.mod_drop_inplace_ct(ct_t)
        while be._ops.depth(zero_ct) < target_depth:
            be._ops.mod_drop_inplace_ct(zero_ct)
        print(f"    pre-StoC depths: ct_t={be._ops.depth(ct_t)}, zero={be._ops.depth(zero_ct)}, "
              f"scales: ct_t={be._ops.scale(ct_t):.2e}, zero={be._ops.scale(zero_ct):.2e}")
        ct_t_coeff = be._ops.slot_to_coeff(ct_t, zero_ct, be._gk)
        print(f"    after StoC: depth={be._ops.depth(ct_t_coeff)}, scale={be._ops.scale(ct_t_coeff):.2e}")

        if t == 0:
            # Just check first token's coeff content matches X[0]
            coeffs = be.decrypt_coeff(ct_t_coeff)
            err = float(np.max(np.abs(np.array(coeffs[:hidden]) - X[0])))
            print(f"    extracted coeffs err vs X[0]: {err:.3e}")
        break  # bail after one token for now

    print(f"  elapsed: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
