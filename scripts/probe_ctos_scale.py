"""Probe: encode e_0 = (1, 0, 0, ...) as coefficients, run CtoS.

If CtoS is the standard inverse canonical embedding, the result should
be a slot vector of all 1.0 (since p(zeta) = 1 for the constant
polynomial). Any other constant tells us the scale factor that's being
applied. NaN tells us about depth/level handling.
"""
from __future__ import annotations

import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    print(f"N={be._N}  num_slots={be._num_slots}  scale=2^50")

    # Test 1: all-ones coefficient vector
    print("\n--- Test 1: c = (1, 1, 1, ..., 1)  (all-ones) ---")
    c = np.ones(be._N)
    ct = be.encrypt_coeff(c.tolist())
    cts = be.coeff_to_slot(ct)
    s0 = np.asarray(be.decrypt(cts[0]))[: be._num_slots]
    s1 = np.asarray(be.decrypt(cts[1]))[: be._num_slots]
    print(f"  out[0] sample: min={s0.min():.4e} max={s0.max():.4e} mean={s0.mean():.4e}")
    print(f"  out[1] sample: min={s1.min():.4e} max={s1.max():.4e} mean={s1.mean():.4e}")

    # Test 2: e_0 (constant = 1)
    print("\n--- Test 2: c = e_0 = (1, 0, 0, ..., 0) ---")
    c = np.zeros(be._N); c[0] = 1.0
    ct = be.encrypt_coeff(c.tolist())
    cts = be.coeff_to_slot(ct)
    s0 = np.asarray(be.decrypt(cts[0]))[: be._num_slots]
    s1 = np.asarray(be.decrypt(cts[1]))[: be._num_slots]
    print(f"  out[0] should be all 1.0 if CtoS is inv canonical embedding")
    print(f"  out[0]: min={s0.min():.4e} max={s0.max():.4e} mean={s0.mean():.4e} std={s0.std():.4e}")
    print(f"  out[1]: min={s1.min():.4e} max={s1.max():.4e} mean={s1.mean():.4e} std={s1.std():.4e}")

    # Compare against a direct slot-encoding sanity (encode then decode)
    print("\n--- Test 3: slot-encode all-ones, decode (sanity, no CtoS) ---")
    pt = be._encoder.encode(be._ctx, [1.0] * be._num_slots, be._scale)
    ct2 = be._encryptor.encrypt(be._ctx, pt)
    s = np.asarray(be.decrypt(ct2))[: be._num_slots]
    print(f"  decoded all-ones: min={s.min():.6e} max={s.max():.6e}")

    # Test 4: e_1 — should give zeta_k for k in [0, N/2)
    print("\n--- Test 4: c = e_1 = (0, 1, 0, ..., 0) ---")
    c = np.zeros(be._N); c[1] = 1.0
    ct = be.encrypt_coeff(c.tolist())
    cts = be.coeff_to_slot(ct)
    s0 = np.asarray(be.decrypt(cts[0]))[: be._num_slots]
    s1 = np.asarray(be.decrypt(cts[1]))[: be._num_slots]
    # Reference: zetas (real and imag) at the standard CKKS slots
    k = np.arange(be._num_slots)
    zetas_real = np.cos(np.pi * (2 * k + 1) / be._N)
    zetas_imag = np.sin(np.pi * (2 * k + 1) / be._N)
    print(f"  out[0]: min={s0.min():.4e} max={s0.max():.4e}  ref-cos: min={zetas_real.min():.4e} max={zetas_real.max():.4e}")
    print(f"  out[1]: min={s1.min():.4e} max={s1.max():.4e}  ref-sin: min={zetas_imag.min():.4e} max={zetas_imag.max():.4e}")
    # Try ratios
    ratio_re = s0[10] / zetas_real[10] if zetas_real[10] != 0 else float("nan")
    print(f"  s0[10]/cos[10] = {ratio_re:.6e}  (should be const if CtoS = inv canonical embedding × const)")


if __name__ == "__main__":
    main()
