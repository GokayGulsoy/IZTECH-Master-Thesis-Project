"""Probe the C++ halevi_shoup_matvec_block kernel directly to find the level mismatch."""
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 6,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    rng = np.random.default_rng(0)
    block = 128
    num_slots = be._num_slots
    x = np.zeros(num_slots)
    x[:block] = rng.standard_normal(block) * 0.1
    ct = be.encrypt(x.tolist())
    print(f"ct depth = {be._ops.depth(ct)}")

    # Encode 2 diagonals at the right depths.
    d0 = list(rng.standard_normal(num_slots) * 0.1)
    d1 = list(rng.standard_normal(num_slots) * 0.1)
    pt_d0 = be._encode(d0)
    pt_d1 = be._encode(d1)
    print(f"  fresh pt d0 depth={be._ops.depth_of_plaintext(pt_d0)}")
    # NOTE: leave diags at depth 0 — C++ kernel mod-drops them per iter.

    lo = [0.0] * num_slots
    hi = [0.0] * num_slots
    for b in range(num_slots // block):
        for j in range(block - 1):
            lo[b * block + j] = 1.0
        for j in range(block - 1, block):
            hi[b * block + j] = 1.0
    pt_lo = be._encode(lo)
    pt_hi = be._encode(hi)
    print(f"  mask depth={be._ops.depth_of_plaintext(pt_lo)}")

    bias_pt = be._encode([0.0] * num_slots)
    be._ops.mod_drop_inplace_pt(bias_pt)
    be._ops.mod_drop_inplace_pt(bias_pt)
    print(f"  bias depth={be._ops.depth_of_plaintext(bias_pt)}")

    be.register_rotation_keys([1, 1 - block])
    print("Calling C++ kernel with shifts=[0,1] ...")
    try:
        out = be._ops.halevi_shoup_matvec_block(
            ct, be._gk, int(block),
            [0, 1], [pt_d0, pt_d1], [pt_lo, pt_lo], [pt_hi, pt_hi],
            False, bias_pt,
        )
        print(f"  OK depth={be._ops.depth(out)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("Calling C++ kernel with shifts=[0] only ...")
    try:
        out = be._ops.halevi_shoup_matvec_block(
            ct, be._gk, int(block),
            [0], [pt_d0], [pt_lo], [pt_hi],
            False, bias_pt,
        )
        print(f"  OK depth={be._ops.depth(out)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("Calling C++ kernel with shifts=[1] only ...")
    try:
        out = be._ops.halevi_shoup_matvec_block(
            ct, be._gk, int(block),
            [1], [pt_d1], [pt_lo], [pt_hi],
            False, bias_pt,
        )
        print(f"  OK depth={be._ops.depth(out)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
