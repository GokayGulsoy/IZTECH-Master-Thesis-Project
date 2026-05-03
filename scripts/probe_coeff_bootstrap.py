"""Investigate coeff-encoded ciphertext bootstrap.

We need to bootstrap a coeff-encoded ct (post-StoC of L1) to reset depth
before L2's CtoS. Tests:
  1. Does regular_bootstrapping accept coeff input?
  2. What depth does bootstrap output (coeff or slot) at?
  3. Can we do CtoS → slot_bootstrap → StoC as a coeff_bootstrap wrapper?
"""
import numpy as np
import time

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
    x = rng.standard_normal(64) * 0.3

    # Test 1: bootstrap a slot ct → check output depth + encoding
    print("\n[Test 1] Bootstrap slot-encoded ct")
    ct_slot = be.encrypt(x.tolist())
    while be._ops.depth(ct_slot) < be._max_depth:
        be._ops.mod_drop_inplace_ct(ct_slot)
    print(f"  pre-bootstrap depth={be._ops.depth(ct_slot)}, encoding={ct_slot.encoding_type()}")
    ct_b = be._ops.regular_bootstrapping(ct_slot, be._gk, be._rk)
    print(f"  post-bootstrap depth={be._ops.depth(ct_b)}, encoding={ct_b.encoding_type()}")
    dec = be.decrypt(ct_b)
    err = float(np.max(np.abs(np.array(dec[:64]) - x)))
    print(f"  err vs original: {err:.3e}")

    # Test 2: bootstrap a coeff ct
    print("\n[Test 2] Bootstrap coeff-encoded ct")
    ct_coeff = be.encrypt_coeff(x.tolist())
    print(f"  pre-bootstrap depth={be._ops.depth(ct_coeff)}, encoding={ct_coeff.encoding_type()}")
    while be._ops.depth(ct_coeff) < be._max_depth:
        be._ops.mod_drop_inplace_ct(ct_coeff)
    try:
        ct_b = be._ops.regular_bootstrapping(ct_coeff, be._gk, be._rk)
        print(f"  post-bootstrap depth={be._ops.depth(ct_b)}, encoding={ct_b.encoding_type()}")
        coeffs = be.decrypt_coeff(ct_b)
        err = float(np.max(np.abs(np.array(coeffs[:64]) - x)))
        print(f"  coeff err: {err:.3e}")
        # Or maybe it returned slot-encoded?
        dec = be.decrypt(ct_b)
        err_slot = float(np.max(np.abs(np.array(dec[:64]) - x)))
        print(f"  slot err: {err_slot:.3e}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 3: Can a freshly-bootstrapped slot ct go through CtoS?
    print("\n[Test 3] After-bootstrap CtoS")
    ct_slot = be.encrypt(x.tolist())
    while be._ops.depth(ct_slot) < be._max_depth:
        be._ops.mod_drop_inplace_ct(ct_slot)
    ct_b = be._ops.regular_bootstrapping(ct_slot, be._gk, be._rk)
    print(f"  post-bootstrap depth={be._ops.depth(ct_b)}")
    if be._ops.depth(ct_b) > 0:
        print(f"  (depth>0; CtoS will reject)")
    try:
        cts = be._ops.coeff_to_slot(ct_b, be._gk)
        print(f"  CtoS succeeded; depth_after={be._ops.depth(cts[0])}")
    except Exception as e:
        print(f"  CtoS rejected: {e}")


if __name__ == "__main__":
    main()
