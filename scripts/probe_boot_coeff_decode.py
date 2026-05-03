"""Hypothesis: regular_bootstrapping on coeff ct returns ct whose polynomial
coefficients still hold our data — only mislabeled as SLOT.
"""
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16, 40-level chain...")
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

    ct = be.encrypt_coeff(x.tolist())
    while be._ops.depth(ct) < be._max_depth:
        be._ops.mod_drop_inplace_ct(ct)
    print(f"  pre-boot: depth={be._ops.depth(ct)}, encoding={ct.encoding_type()}")

    ct_b = be._ops.regular_bootstrapping(ct, be._gk, be._rk)
    print(f"  post-boot: depth={be._ops.depth(ct_b)}, encoding={ct_b.encoding_type()}")

    # KEY TEST: decode as COEFFICIENT despite SLOT label
    pt = be._decryptor.decrypt(be._ctx, ct_b)
    coeffs = be._encoder.decode_coeff(pt)
    err_coeff = float(np.max(np.abs(np.array(coeffs[:64]) - x)))
    print(f"\n  COEFFICIENT decode err vs x: {err_coeff:.3e}  ← THE TEST")

    # Also check slot decode (should be FFT of x)
    slots = be._encoder.decode(pt)
    print(f"  slot[:8] = {np.real(slots[:8]) if hasattr(slots[0], 'imag') else slots[:8]}")
    print(f"  expected x[:8] = {x[:8]}")

    if err_coeff < 1e-2:
        print("\n  ✅ HYPOTHESIS CONFIRMED — coeff data round-trips through bootstrap!")
        print("     We can chain: coeff_matvec → CtoS → polyval → StoC → bootstrap → coeff_matvec")
    else:
        # Maybe scaled or sign-flipped
        print(f"\n  ratio coeffs[:8]/x[:8] = {np.array(coeffs[:8])/x[:8]}")


if __name__ == "__main__":
    main()
