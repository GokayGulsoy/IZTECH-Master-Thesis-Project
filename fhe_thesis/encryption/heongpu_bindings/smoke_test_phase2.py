"""Phase 2 smoke tests for the HEonGPU pybind11 wrapper.

Exercises the two ops that gate every higher-level kernel:

  1. ``rotate_rows_inplace`` — left-cyclic SIMD rotation, used by
     Halevi-Shoup matmul, sum-slots reductions, and our matrix-packed
     attention.
  2. ``regular_bootstrapping`` — depth refresh, the load-bearing op for
     LPAN inference (depth ~33/layer).

Run from the repo root so the package import resolves:

    python -m fhe_thesis.encryption.heongpu_bindings.smoke_test_phase2
"""

from __future__ import annotations

import sys
import time

from . import (
    CKKSContext,
    Decryptor,
    Encoder,
    Encryptor,
    KeyGenerator,
    Operator,
)


def test_rotate() -> bool:
    print("\n=== Phase 2a: rotate_rows ===")
    ctx = CKKSContext(8192, [60, 30, 30, 30], [60])
    kg = KeyGenerator(ctx)
    sk = kg.generate_secret_key(ctx)
    pk = kg.generate_public_key(ctx, sk)
    rk = kg.generate_relin_key(ctx, sk)
    gk = kg.generate_galois_key(ctx, sk, [1, 2, 4, 8, 16])  # power-of-two shifts

    enc = Encoder(ctx)
    encryptor = Encryptor(ctx, pk)
    decryptor = Decryptor(ctx, sk)
    ops = Operator(ctx, enc)

    slots = ctx.slot_count
    msg = [float(i) for i in range(slots)]
    ct = encryptor.encrypt(ctx, enc.encode(ctx, msg, 2.0**30))

    for shift in (1, 2, 4, 8, 16):
        # Re-encrypt for each shift so we test from a clean state.
        ct = encryptor.encrypt(ctx, enc.encode(ctx, msg, 2.0**30))
        ops.rotate_rows_inplace(ct, gk, shift)
        out = enc.decode(decryptor.decrypt(ctx, ct))
        # Expect msg shifted left by `shift` (HEonGPU rotate_rows convention).
        ok = all(abs(out[i] - msg[i + shift]) < 1e-1 for i in range(8))
        print(f"  shift={shift:2d}: out[0:5]={[round(x, 2) for x in out[:5]]}  "
              f"expected~{msg[shift:shift + 5]}  ->  {'OK' if ok else 'FAIL'}")
        if not ok:
            return False
    return True


def test_bootstrap() -> bool:
    print("\n=== Phase 2b: regular_bootstrapping ===")
    # Use the same params as HEonGPU's bootstrap example (insecure but functional).
    ctx = CKKSContext(
        4096,
        [60] + [50] * 30,        # Q_bits
        [60, 60, 60],            # P_bits
        sec_none=True,
    )
    scale = 2.0**50

    kg = KeyGenerator(ctx)
    sk = kg.generate_secret_key(ctx)
    pk = kg.generate_public_key(ctx, sk)
    rk = kg.generate_relin_key(ctx, sk)

    enc = Encoder(ctx)
    encryptor = Encryptor(ctx, pk)
    decryptor = Decryptor(ctx, sk)
    ops = Operator(ctx, enc)

    # Boot params + galois keys for bootstrapping shifts.
    ops.generate_bootstrapping_params(scale, CtoS_piece=3, StoC_piece=3,
                                      taylor_number=11, less_key_mode=True)
    boot_shifts = ops.bootstrapping_key_indexs()
    print(f"  bootstrapping needs {len(boot_shifts)} galois shifts")
    gk = kg.generate_galois_key(ctx, sk, boot_shifts)

    msg = [0.2] * ctx.slot_count
    ct = encryptor.encrypt(ctx, enc.encode(ctx, msg, scale))

    # Drain the modulus chain down to a single level (matches example).
    for _ in range(29):
        ops.mod_drop_inplace_ct(ct)
    print(f"  depth before bootstrap: {ops.depth(ct)}")

    t0 = time.perf_counter()
    ct_boot = ops.regular_bootstrapping(ct, gk, rk)
    dt = time.perf_counter() - t0
    print(f"  bootstrap wall: {dt * 1000:.1f} ms")
    print(f"  depth after bootstrap:  {ops.depth(ct_boot)}")

    out = enc.decode(decryptor.decrypt(ctx, ct_boot))
    sample = [round(out[i], 4) for i in range(5)]
    print(f"  decoded[0:5]={sample}  expected~[0.2, 0.2, 0.2, 0.2, 0.2]")
    return all(abs(out[i] - 0.2) < 5e-2 for i in range(64))


def main() -> int:
    if not test_rotate():
        print("\nPhase 2 FAILED at rotate")
        return 1
    if not test_bootstrap():
        print("\nPhase 2 FAILED at bootstrap")
        return 2
    print("\nPhase 2 smoke tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
