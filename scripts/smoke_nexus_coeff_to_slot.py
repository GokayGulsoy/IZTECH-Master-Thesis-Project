"""Smoke test: NEXUS coeff_matvec → solo_coeff_to_slot → slot domain.

Verifies that we can convert a coefficient-encoded ciphertext (the
output of :meth:`HEonGPUBackend.coeff_matvec`) into a slot-encoded
ciphertext via HEonGPU's `solo_coeff_to_slot` so that downstream
slot-domain ops (polyval, mul, etc.) can consume the result without
the decrypt-then-re-encrypt cheat.

Required prerequisite: `configure_bootstrapping()` (populates the
BSGS matrices that `solo_coeff_to_slot` reads from internally and
generates the rotation Galois keys).
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main() -> int:
    in_dim, out_dim = 128, 256  # m·n = 32768 = N/2 — fits one solo_coeff_to_slot
    print(f"shape: in={in_dim} out={out_dim}  m·n = {in_dim * out_dim}")

    print("\nInit HEonGPU N=2^16 with bootstrap-ready chain...")
    t0 = time.time()
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    print(f"  ctx init: {time.time() - t0:.2f}s   N={be._N}")

    t0 = time.time()
    print("Configuring bootstrapping (populates CtoS matrices)...")
    be.configure_bootstrapping()
    print(f"  configure_bootstrapping: {time.time() - t0:.2f}s")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim).astype(np.float64) * 0.3
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float64) * 0.05
    expected = W @ x

    print("\n[1] coeff_matvec...")
    t = time.time()
    ct_x_coeff = be.encrypt_coeff(x.tolist())
    ct_y_coeff = be.coeff_matvec(ct_x_coeff, W, in_dim=in_dim)
    print(f"  wall: {time.time() - t:.3f}s")

    # Sanity: decrypt coefficients, check the predicted positions.
    coeffs = np.asarray(be.decrypt_coeff(ct_y_coeff))
    extracted = np.array([coeffs[(i + 1) * in_dim - 1] for i in range(out_dim)])
    err = np.max(np.abs(extracted - expected))
    print(f"  coeff-domain extract max-err: {err:.3e}")

    print("\n[2] solo_coeff_to_slot (homomorphic CtoS)...")
    t = time.time()
    ct_y_slot = be.coeff_to_slot(ct_y_coeff)
    print(f"  wall: {time.time() - t:.3f}s")

    # Decode in slot domain.
    slots = np.asarray(be.decrypt(ct_y_slot))
    print(f"  slot vector len: {len(slots)}")

    # The polynomial coefficients are at slot indices 0..N/2-1.
    # Our values of interest live at indices [n-1, 2n-1, ..., m·n - 1].
    extracted_slots = np.array([slots[(i + 1) * in_dim - 1] for i in range(out_dim)])
    err_slot = np.max(np.abs(extracted_slots - expected))
    print(f"  slot-domain extract max-err: {err_slot:.3e}")

    if err_slot < 5e-2:
        print("\n  PASS — coeff→slot conversion works on H100")
        return 0
    print("\n  FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
