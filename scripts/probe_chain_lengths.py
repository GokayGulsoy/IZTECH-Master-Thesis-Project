"""Try bootstrap with longer modulus chain to see if post-boot depth allows CtoS."""
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def try_chain(num_levels, hw=16):
    print(f"\n=== q_prime chain with {num_levels} levels, hw={hw} ===")
    try:
        be = HEonGPUBackend(
            poly_modulus_degree=1 << 16,
            q_prime_bits=(60,) + (50,) * num_levels,
            p_prime_bits=(60, 60, 60),
            scale_bits=50,
            bootstrap_hamming_weight=hw,
            sec_none=True,
        )
        be.configure_bootstrapping()
        print(f"  init OK, max_depth={be._max_depth}")
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64) * 0.3
        ct = be.encrypt(x.tolist())
        while be._ops.depth(ct) < be._max_depth:
            be._ops.mod_drop_inplace_ct(ct)
        ct_b = be._ops.regular_bootstrapping(ct, be._gk, be._rk)
        post_depth = be._ops.depth(ct_b)
        free = be._max_depth - post_depth
        print(f"  post-boot depth={post_depth}, free levels={free}")
        if free >= 10:
            print(f"  ✓ ENOUGH FOR CtoS+linear+CtoS chain")
            # Try CtoS
            try:
                cts = be._ops.coeff_to_slot(ct_b, be._gk)
                print(f"  ✓ CtoS succeeded post-boot")
            except Exception as e:
                print(f"  ✗ CtoS rejected: {e}")
        else:
            print(f"  ✗ NOT ENOUGH: need 10+ free levels for chain")
    except Exception as e:
        print(f"  init failed: {e}")


def main():
    for n in [30, 40, 50]:
        try_chain(n)


if __name__ == "__main__":
    main()
