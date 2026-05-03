"""Test if regular_bootstrapping on a coeff ct preserves data (treating it as
coefficients) and outputs a slot ct ready for polyval.

If yes → we can chain by doing:
  coeff_matvec → CtoS → polyval → StoC → mod_drop → bootstrap → slot output.
But then the output is slot, can't feed next coeff_matvec without StoC,
which needs depth=0. So this is the diagnostic.
"""
import numpy as np
import time

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16 with 40-level chain...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 40,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    print(f"  max_depth={be._max_depth}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(64) * 0.3

    # Step 1: encrypt as coeffs
    ct = be.encrypt_coeff(x.tolist())
    print(f"\nstep1 coeff_encrypt: depth={be._ops.depth(ct)}, encoding={ct.encoding_type()}")

    # Mod_drop to max
    while be._ops.depth(ct) < be._max_depth:
        be._ops.mod_drop_inplace_ct(ct)
    print(f"step2 mod_dropped:   depth={be._ops.depth(ct)}, encoding={ct.encoding_type()}")

    # Bootstrap
    t0 = time.time()
    ct_b = be._ops.regular_bootstrapping(ct, be._gk, be._rk)
    t_boot = time.time() - t0
    print(f"step3 post-bootstrap: depth={be._ops.depth(ct_b)}, encoding={ct_b.encoding_type()}, wall={t_boot*1000:.0f}ms")

    # Try decrypt as slot
    dec_slot = be.decrypt(ct_b)
    err_slot_first = float(np.max(np.abs(np.array(dec_slot[:64]) - x)))
    print(f"  slot[:64] err vs x: {err_slot_first:.3e}")

    # Try with bit-rev permutation (since coeff[i] → slot[bitrev(i)])
    log_n = (be._N // 2).bit_length() - 1
    def bitrev(i, bits=log_n):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r
    err_slot_brev = float(np.max(np.abs(np.array([dec_slot[bitrev(i)] for i in range(64)]) - x)))
    print(f"  slot[bitrev(i)] err vs x: {err_slot_brev:.3e}")

    if err_slot_brev < 1e-2:
        print("  ✓ bootstrap of coeff ct DOES place data in slot[bitrev(i)] form — usable!")
    elif err_slot_first < 1e-2:
        print("  ✓ bootstrap of coeff ct DOES place data in slot[i] form")
    else:
        # Maybe data is scaled
        print(f"  bootstrap output mag[:8]: {np.abs(dec_slot[:8])}")
        print(f"  expected x[:8]: {np.abs(x[:8])}")


if __name__ == "__main__":
    main()
