"""Phase 1 smoke test for the HEonGPU pybind11 wrapper.

Mirrors HEonGPU's example ``2_basic_ckks.cpp``: encrypt a vector,
square it homomorphically, multiply by a plaintext and add to itself,
then verify the decoded result matches the expected pattern.

Run from the directory that contains the built ``_heongpu*.so``:

    python -m fhe_thesis.encryption.heongpu_bindings.smoke_test
"""

from __future__ import annotations

import math
import sys

from . import (
    CKKSContext,
    Decryptor,
    Encoder,
    Encryptor,
    KeyGenerator,
    Operator,
)


def _approx(actual: float, expected: float, tol: float = 1e-2) -> bool:
    return abs(actual - expected) <= tol * max(1.0, abs(expected))


def main() -> int:
    poly_n = 8192
    ctx = CKKSContext(poly_n, [60, 30, 30, 30], [60])
    ctx.print_parameters()

    keygen = KeyGenerator(ctx)
    sk = keygen.generate_secret_key(ctx)
    pk = keygen.generate_public_key(ctx, sk)
    rk = keygen.generate_relin_key(ctx, sk)

    encoder = Encoder(ctx)
    encryptor = Encryptor(ctx, pk)
    decryptor = Decryptor(ctx, sk)
    ops = Operator(ctx, encoder)

    scale = 2.0**30
    slots = ctx.slot_count
    msg = [3.0] * slots
    msg[0:5] = [10.0, 20.0, 30.0, 40.0, 0.5]

    p1 = encoder.encode(ctx, msg, scale)
    c1 = encryptor.encrypt(ctx, p1)

    # Square in-place.
    ops.multiply_inplace(c1, c1)
    ops.relinearize_inplace(c1, rk)
    ops.rescale_inplace(c1)

    p2 = decryptor.decrypt(ctx, c1)
    sq = encoder.decode(p2)
    print("After square (first 8):", [round(v, 3) for v in sq[:8]])

    expected_sq = [100.0, 400.0, 900.0, 1600.0, 0.25, 9.0, 9.0, 9.0]
    for i, exp in enumerate(expected_sq):
        if not _approx(sq[i], exp, tol=1e-2):
            print(f"FAIL: sq[{i}]={sq[i]} expected~{exp}")
            return 1

    # multiply_plain by 0.25 then add to itself -> *0.5
    p3 = encoder.encode(ctx, [0.25] * slots, scale)
    ops.mod_drop_inplace_pt(p3)
    ops.multiply_plain_inplace(c1, p3)
    # add to self via second copy: easiest is encrypt-decrypt-roundtrip;
    # here we just call add_inplace(c1, c1) since HEonGPU supports aliasing.
    ops.add_inplace(c1, c1)
    ops.rescale_inplace(c1)

    p4 = decryptor.decrypt(ctx, c1)
    out = encoder.decode(p4)
    print("After *0.5 (first 8):", [round(v, 3) for v in out[:8]])

    expected_out = [50.0, 200.0, 450.0, 800.0, 0.125, 4.5, 4.5, 4.5]
    for i, exp in enumerate(expected_out):
        if not _approx(out[i], exp, tol=2e-2):
            print(f"FAIL: out[{i}]={out[i]} expected~{exp}")
            return 1

    print("\nPhase 1 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
