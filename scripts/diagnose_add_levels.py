"""Test add between ciphertexts at different chain levels."""
from __future__ import annotations
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main() -> int:
    be = HEonGPUBackend(
        poly_modulus_degree=32768,
        q_prime_bits=[60] + [40] * 18,
        p_prime_bits=[60],
        scale_bits=40,
    )
    n = be.capabilities.n_slots
    a = [1.0, 2.0, 3.0, 4.0] + [0.0] * (n - 4)
    b = [10.0, 20.0, 30.0, 40.0] + [0.0] * (n - 4)
    ones = [1.0] * n

    ca = be.encrypt(a)                # depth 0
    cb = be.encrypt(b)                # depth 0
    cb_mul = be.mul_plain(cb, ones)   # depth 1 (rescaled), value = b

    # Sanity: cb_mul should still equal b
    out_b = be.decrypt(cb_mul)
    print(f"cb_mul[:4]      = {out_b[:4]}     (expected [10,20,30,40])")

    # Now add ca (depth 0) + cb_mul (depth 1). Expected = a + b = [11,22,33,44].
    csum = be.add(ca, cb_mul)
    out = be.decrypt(csum)
    print(f"add(ca,cb_mul)[:4] = {out[:4]}  (expected [11,22,33,44])")

    # Reverse order
    csum2 = be.add(cb_mul, ca)
    out2 = be.decrypt(csum2)
    print(f"add(cb_mul,ca)[:4] = {out2[:4]}  (expected [11,22,33,44])")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
