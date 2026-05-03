"""Validate the HEonGPU coeff_to_slot binding against numpy reference.

`coeff_to_slot` is the inverse canonical embedding: it takes a polynomial
``p(X) = sum c_i X^i`` (encoded as plaintext coefficients) and returns 2
slot-encoded ciphertexts such that

    out[0].slots[k] = real( p(zeta_k) )
    out[1].slots[k] = imag( p(zeta_k) )

where ``zeta_k`` are the primitive 2N-th roots of unity used by the CKKS
canonical embedding (slot index k ↦ zeta_k = exp(i·pi·(2k+1)/N)).

This script:
  1. Sample random real coefficients c.
  2. Compute reference slots via numpy: p(zeta_k) for k in [0, N/2).
  3. Encode c as coefficients, encrypt, run CtoS, decrypt, compare.

If err is small (~1e-5), the binding semantics match the standard CKKS
canonical embedding and we can build the slot-extraction kernel on top.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def canonical_embedding_slots(coeffs: np.ndarray, N: int) -> np.ndarray:
    """Reference: p(zeta_k) for slot k in [0, N/2)."""
    n_slots = N // 2
    # Standard CKKS uses zeta_k = exp(i * pi * (2k + 1) / N) bit-reversed,
    # but we'll just check magnitudes / patterns — the bit-rev only
    # permutes the slots.
    k = np.arange(n_slots)
    zetas = np.exp(1j * np.pi * (2 * k + 1) / N)
    # Vandermonde-style evaluation: p(zeta_k) = sum_i c_i * zeta_k^i
    powers = zetas[:, None] ** np.arange(N)[None, :]  # (n_slots, N)
    return powers @ coeffs


def main() -> int:
    print("Init HEonGPU N=2^16 with bootstrap-ready chain...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    print(f"  N={be._N}  num_slots={be._num_slots}")

    print("Configuring bootstrapping (populates CtoS matrices)...")
    t = time.time()
    be.configure_bootstrapping()
    print(f"  configure_bootstrapping: {time.time() - t:.2f}s")

    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(be._N).astype(np.float64) * 0.05

    # --- numpy reference ---
    ref = canonical_embedding_slots(coeffs, be._N)
    print(f"\nReference (canonical embedding):")
    print(f"  |real|: min={np.min(np.abs(ref.real)):.3e}  max={np.max(np.abs(ref.real)):.3e}")
    print(f"  |imag|: min={np.min(np.abs(ref.imag)):.3e}  max={np.max(np.abs(ref.imag)):.3e}")

    # --- HEonGPU CtoS ---
    print("\nEncrypt + CtoS at depth 0...")
    ct = be.encrypt_coeff(coeffs.tolist())
    t = time.time()
    cts = be.coeff_to_slot(ct)
    print(f"  CtoS wall: {time.time() - t:.3f}s   {len(cts)} cts")

    s_re = np.asarray(be.decrypt(cts[0]))[: be._num_slots]
    s_im = np.asarray(be.decrypt(cts[1]))[: be._num_slots]

    # Try direct comparison first
    err_re_direct = np.max(np.abs(s_re - ref.real))
    err_im_direct = np.max(np.abs(s_im - ref.imag))
    print(f"\nDirect compare:")
    print(f"  real err: {err_re_direct:.3e}")
    print(f"  imag err: {err_im_direct:.3e}")

    # Try bit-reversed permutation
    n_slots = be._num_slots
    log_n = int(np.log2(n_slots))

    def bitrev(i, bits):
        r = 0
        for b in range(bits):
            r = (r << 1) | ((i >> b) & 1)
        return r

    perm = np.array([bitrev(i, log_n) for i in range(n_slots)])
    err_re_br = np.max(np.abs(s_re - ref.real[perm]))
    err_im_br = np.max(np.abs(s_im - ref.imag[perm]))
    print(f"\nBit-reversed compare:")
    print(f"  real err: {err_re_br:.3e}")
    print(f"  imag err: {err_im_br:.3e}")

    # Try with the 5^k indexing (CKKS standard)
    five_pow = np.array([pow(5, i, 2 * be._N) for i in range(n_slots)])
    # Map slot k → which root of unity it represents
    # Standard CKKS: slot k uses zeta^{5^k} where zeta = exp(2*pi*i / 2N)
    zeta_2N = np.exp(2j * np.pi / (2 * be._N))
    powers_5k = zeta_2N ** five_pow
    powers_mat = powers_5k[:, None] ** np.arange(be._N)[None, :]
    ref_5k = powers_mat @ coeffs
    err_re_5k = np.max(np.abs(s_re - ref_5k.real))
    err_im_5k = np.max(np.abs(s_im - ref_5k.imag))
    print(f"\n5^k indexing compare (standard CKKS):")
    print(f"  real err: {err_re_5k:.3e}")
    print(f"  imag err: {err_im_5k:.3e}")

    best = min(err_re_direct, err_re_br, err_re_5k)
    if best < 1e-3:
        print(f"\n  PASS — binding semantics confirmed (best err {best:.3e})")
        return 0
    print(f"\n  Need to investigate slot ordering further (best {best:.3e})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
