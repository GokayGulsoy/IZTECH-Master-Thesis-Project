"""Test whether mod_drop_inplace_pt produces a plaintext that can multiply
a ciphertext arrived at depth>0 via RESCALE (not via ct mod_drop)."""
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
    N = be._num_slots
    x = rng.standard_normal(N) * 0.1

    print("=== Test 1: mul -> rescale -> mul (mod-drop pt) ===")
    ct = be.encrypt(x.tolist())
    print(f"  initial ct depth={be._ops.depth(ct)}")
    pt1 = be._encode([0.5] * N)
    be._ops.multiply_plain_inplace(ct, pt1)
    print(f"  after mul: depth={be._ops.depth(ct)}")
    be._ops.rescale_inplace(ct)
    print(f"  after rescale: depth={be._ops.depth(ct)}")
    pt2 = be._encode([0.3] * N)
    print(f"  pt2 fresh depth={be._ops.depth_of_plaintext(pt2)}")
    while be._ops.depth_of_plaintext(pt2) < be._ops.depth(ct):
        be._ops.mod_drop_inplace_pt(pt2)
    print(f"  pt2 after drops depth={be._ops.depth_of_plaintext(pt2)}")
    try:
        be._ops.multiply_plain_inplace(ct, pt2)
        print(f"  OK! depth={be._ops.depth(ct)}")
    except Exception as e:
        print(f"  FAIL: {e}")

    print()
    print("=== Test 2: ct mod_drop -> mul (mod-drop pt to match) ===")
    ct = be.encrypt(x.tolist())
    print(f"  initial ct depth={be._ops.depth(ct)}")
    be._ops.mod_drop_inplace_ct(ct)
    print(f"  after ct mod_drop: depth={be._ops.depth(ct)}")
    pt = be._encode([0.5] * N)
    while be._ops.depth_of_plaintext(pt) < be._ops.depth(ct):
        be._ops.mod_drop_inplace_pt(pt)
    print(f"  pt after drops depth={be._ops.depth_of_plaintext(pt)}")
    try:
        be._ops.multiply_plain_inplace(ct, pt)
        print(f"  OK! depth={be._ops.depth(ct)}")
    except Exception as e:
        print(f"  FAIL: {e}")


if __name__ == "__main__":
    main()
