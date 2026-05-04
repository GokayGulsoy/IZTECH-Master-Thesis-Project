"""Phase 8a smoke: multiply_power_of_x parity vs hand-computed coeff shift.

Encode a polynomial p(x) = sum c_j x^j via encode_coeff (NEXUS-style
coefficient packing), encrypt, multiply by x^k, decrypt, decode, and
check that the resulting coefficients match the negacyclic shift
applied to the original coefficient vector.

Negacyclic shift of c by k: c'[i] = c[(i - k) mod N], with sign flip
if (i - k) wrapped (under x^N = -1).
"""
from __future__ import annotations

import numpy as np

from fhe_thesis.encryption.heongpu_backend import (
    HeonGPUBackend, BackendConfig
)


def negacyclic_shift_ref(coeffs: np.ndarray, k: int) -> np.ndarray:
    """Reference: multiply polynomial by x^k mod (x^N + 1) on coefficients."""
    N = len(coeffs)
    out = np.zeros(N, dtype=coeffs.dtype)
    k_mod = k % (2 * N)
    for i in range(N):
        s = (i - k_mod) % (2 * N)
        wrap = 1 if s >= N else 0
        s = s % N
        out[i] = -coeffs[s] if wrap else coeffs[s]
    return out


def main() -> None:
    cfg = BackendConfig(
        n_slots=2**14,                      # smaller ring for fast test
        q_bits=(60, 50, 50, 50),
        p_bits=(60, 60, 60),
        scale=2**50,
        sec_none=True,
        max_depth=2,
    )
    be = HeonGPUBackend(cfg)
    N = cfg.n_slots                          # we use n_slots as ring degree

    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=N).astype(np.float64) * 0.1

    # encode_coeff: encode as polynomial coefficients (not slots).
    pt = be._ops_encoder.encode_coeff(be._ctx, list(coeffs), float(cfg.scale))
    ct = be._ops_encryptor.encrypt(pt)

    # Decode roundtrip sanity:
    dec = be._ops_decryptor.decrypt(ct)
    rec = be._ops_encoder.decode_coeff(dec)
    rec = np.array(rec[:N])
    err0 = np.max(np.abs(rec - coeffs))
    print(f"roundtrip err (k=0): {err0:.3e}")

    for k in [1, 17, N // 2 - 1, N, N + 5, 2 * N - 1, -3]:
        # Fresh ct each time (multiply_power_of_x is in-place).
        pt2 = be._ops_encoder.encode_coeff(
            be._ctx, list(coeffs), float(cfg.scale)
        )
        ct2 = be._ops_encryptor.encrypt(pt2)
        be._ops.multiply_power_of_x_inplace(ct2, k)
        dec2 = be._ops_decryptor.decrypt(ct2)
        rec2 = np.array(be._ops_encoder.decode_coeff(dec2)[:N])

        ref = negacyclic_shift_ref(coeffs, k)
        err = np.max(np.abs(rec2 - ref))
        print(f"k={k:6d}  err={err:.3e}  "
              f"sample[:4]={rec2[:4].round(4).tolist()}  "
              f"ref[:4]={ref[:4].round(4).tolist()}")
        assert err < 1e-3, f"Large err at k={k}: {err}"

    print("PASS")


if __name__ == "__main__":
    main()
