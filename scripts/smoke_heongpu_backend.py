"""Quick smoke test for HEonGPUBackend basic ops."""
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
    print(f"slots={be.capabilities.n_slots}  scale={be.scale}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(64) * 0.5
    y = rng.standard_normal(64) * 0.3

    # encrypt + decrypt
    cx = be.encrypt(x)
    out = be.decrypt(cx)
    err = np.max(np.abs(np.asarray(out[:64]) - x))
    print(f"encrypt+decrypt err: {err:.4g}")

    # mul_plain
    cmul = be.mul_plain(cx, list(y) + [0.0] * (be.capabilities.n_slots - 64))
    out = be.decrypt(cmul)
    err = np.max(np.abs(np.asarray(out[:64]) - x * y))
    print(f"mul_plain err:        {err:.4g}   expected~{(x * y)[:3]}")
    print(f"got: {out[:3]}")

    # add
    cy = be.encrypt(y)
    cadd = be.add(cx, cy)
    out = be.decrypt(cadd)
    err = np.max(np.abs(np.asarray(out[:64]) - (x + y)))
    print(f"add err:              {err:.4g}")

    # rotate(1)
    crot = be.rotate(cx, 1)
    out = be.decrypt(crot)
    err = np.max(np.abs(np.asarray(out[:63]) - x[1:64]))
    print(f"rotate(1) err:        {err:.4g}")

    # rotate(5)
    crot = be.rotate(cx, 5)
    out = be.decrypt(crot)
    err = np.max(np.abs(np.asarray(out[:59]) - x[5:64]))
    print(f"rotate(5) err:        {err:.4g}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
