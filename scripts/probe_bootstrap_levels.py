"""Probe actual CtoS_level_ / StoC_level_ values + bootstrap depth budget."""
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    for n_levels in [30, 35, 40]:
        for ctos_piece in [3, 4, 5]:
            try:
                print(f"\n=== chain={n_levels}, CtoS_piece={ctos_piece}, StoC_piece={ctos_piece} ===")
                be = HEonGPUBackend(
                    poly_modulus_degree=1 << 16,
                    q_prime_bits=(60,) + (50,) * n_levels,
                    p_prime_bits=(60, 60, 60),
                    scale_bits=50,
                    bootstrap_hamming_weight=16,
                    sec_none=True,
                )
                be.configure_bootstrapping(CtoS_piece=ctos_piece, StoC_piece=ctos_piece)
                print(f"  CtoS_level_={be._ops.coeff_to_slot_level()}")
                print(f"  StoC_level_={be._ops.slot_to_coeff_level()}")
                print(f"  max_depth={be._max_depth}")
                # Bootstrap a slot ct, measure post-boot depth
                rng = np.random.default_rng(0)
                x = rng.standard_normal(64) * 0.3
                ct = be.encrypt(x.tolist())
                while be._ops.depth(ct) < be._max_depth:
                    be._ops.mod_drop_inplace_ct(ct)
                ct_b = be._ops.regular_bootstrapping(ct, be._gk, be._rk)
                print(f"  post-boot depth={be._ops.depth(ct_b)}")
                print(f"  free levels post-boot = {be._max_depth - be._ops.depth(ct_b)}")
                # Can we CtoS this output? CtoS needs depth == CtoS_level_
                ctos_lvl = be._ops.coeff_to_slot_level()
                if be._ops.depth(ct_b) <= ctos_lvl:
                    print(f"  ✓ post-boot depth ≤ CtoS_level → CAN chain CtoS after boot")
                else:
                    gap = be._ops.depth(ct_b) - ctos_lvl
                    print(f"  ✗ gap {gap} levels — boot output too deep for CtoS")
                del be
            except Exception as e:
                print(f"  failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
