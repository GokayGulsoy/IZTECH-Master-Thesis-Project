"""Smoke test for NEXUS-style coefficient encoding in HEonGPU.

Validates:
1. encode_coeff / decode_coeff round-trip.
2. ct (encrypted from coeff-encoded pt) × pt (slot-encoded) — what does
   this multiplication mean? (Should still be a valid CKKS op since both
   live in the NTT domain; the result interpretation depends on encoding.)
3. ct (coeff-encrypted x) × pt (coeff-encoded W) — produces a polynomial
   whose coefficients are the negacyclic convolution. This is the
   foundation of NEXUS-style coefficient-packed matmul.

For a polynomial of degree N (with slot count N/2), if:
  pt_x has coefficients [x_0, x_1, ..., x_{n-1}, 0, ..., 0]   (length N)
  pt_W has coefficients [W_0, W_1, ..., W_{m-1}, 0, ..., 0]
then their product polynomial pt_x · pt_W mod (X^N + 1) has coefficients
that are the linear (or for sparse, equivalent to linear) convolution.

Specifically, if we set:
  pt_x = Σ x_i X^i,  pt_W = Σ w_{n-1-j} X^j  (W reversed)
then coeff of X^{n-1} in (pt_x · pt_W) = Σ_i x_i · w_i = ⟨x, W⟩
"""
from __future__ import annotations

import numpy as np

from fhe_thesis.encryption import heongpu_bindings as hg


def main() -> int:
    N = 1 << 14  # degree
    slots = N // 2
    print(f"Init HEonGPU N={N} slots={slots}")

    ctx = hg._heongpu.CKKSContext(
        poly_modulus_degree=N,
        q_bits=[60] + [40] * 4 + [60],
        p_bits=[60],
        sec_none=True,
    )
    encoder = hg._heongpu.Encoder(ctx)
    keygen = hg._heongpu.KeyGenerator(ctx)
    sk = keygen.generate_secret_key(ctx)
    pk = keygen.generate_public_key(ctx, sk)
    rk = keygen.generate_relin_key(ctx, sk)
    encryptor = hg._heongpu.Encryptor(ctx, pk)
    decryptor = hg._heongpu.Decryptor(ctx, sk)
    ops = hg._heongpu.Operator(ctx, encoder)

    scale = float(2**40)
    rng = np.random.default_rng(0)

    # ── Test 1: coefficient encode round-trip ─────────────────────
    print("\n[1] coeff encode/decode round-trip")
    n = 32
    x = rng.standard_normal(n).astype(np.float64) * 0.3
    x_padded = np.zeros(N, dtype=np.float64)
    x_padded[:n] = x
    pt = encoder.encode_coeff(ctx, x_padded.tolist(), scale)
    decoded = np.asarray(encoder.decode_coeff(pt))
    err = np.max(np.abs(decoded[:n] - x))
    print(f"  max-err (first {n}) = {err:.3e}")
    assert err < 1e-5, "coeff round-trip broken"

    # ── Test 2: encrypt + decrypt coeff-encoded ──────────────────
    print("\n[2] encrypt/decrypt coeff-encoded plaintext")
    ct = encryptor.encrypt(ctx, pt)
    pt_dec = decryptor.decrypt(ctx, ct)
    decoded2 = np.asarray(encoder.decode_coeff(pt_dec))
    err = np.max(np.abs(decoded2[:n] - x))
    print(f"  max-err (first {n}) = {err:.3e}")
    assert err < 1e-3, "coeff encrypt round-trip broken"

    # ── Test 3: coefficient-domain polynomial multiplication ────
    # Encode x as Σ x_i X^i, w as Σ w_{n-1-j} X^j (reversed),
    # multiply → coefficient of X^{n-1} should equal Σ x_i w_i.
    # This is one inner product per multiplication.
    print("\n[3] coefficient-packed inner product (1 mul = 1 dot)")
    w = rng.standard_normal(n).astype(np.float64) * 0.3
    w_rev_padded = np.zeros(N, dtype=np.float64)
    w_rev_padded[:n] = w[::-1]  # encode reversed in low coeffs
    pt_w = encoder.encode_coeff(ctx, w_rev_padded.tolist(), scale)

    ops.multiply_plain_inplace(ct, pt_w)
    ops.rescale_inplace(ct)
    pt_out = decryptor.decrypt(ctx, ct)
    out = np.asarray(encoder.decode_coeff(pt_out))
    expected_dot = float(np.dot(x, w))
    got = out[n - 1]
    err = abs(got - expected_dot)
    print(f"  expected ⟨x,w⟩ = {expected_dot:+.6f}")
    print(f"  got   coeff[{n - 1}] = {got:+.6f}")
    print(f"  err = {err:.3e}")
    if err < 1e-2:
        print("  PASS — coefficient-packed inner product works")
    else:
        print("  FAIL")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
