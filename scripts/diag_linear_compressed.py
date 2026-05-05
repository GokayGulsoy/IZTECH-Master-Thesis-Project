"""Diagnose linear_compressed correctness.

Test 1: encode_coeff([c, 0,...,0]) → decrypt → decode. What do we see?
Test 2: encode_coeff(x_padded) * encode_coeff(w_inner) → decode. Coeff[0]?
Test 3: decode_coeff if available.
"""
from __future__ import annotations

import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import (
    enc_compress, _encode_weight_row_inner_product,
)


def main():
    N = 1 << 16
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 6,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    print(f"N={be._N}, num_slots={be._num_slots}")

    # ----- Test 1: constant polynomial c = 0.5 -----
    print("\n=== Test 1: poly p(x) = 0.5 (only coeff 0) ===")
    pt = be._encoder.encode_coeff(be._ctx, [0.5] + [0.0] * (N - 1), be._scale)
    ct = be._encryptor.encrypt(be._ctx, pt)
    ptd = be._decryptor.decrypt(be._ctx, ct)
    decoded = np.array(be._encoder.decode(ptd))
    print(f"  decoded[:8] = {decoded[:8]}")
    print(f"  decoded.mean() = {decoded.mean():.6f}")
    print(f"  decoded.sum() = {decoded.sum():.6f}")
    print(f"  expected (slot view of constant poly 0.5): all slots = 0.5")

    # ----- Test 2: encode_coeff(x_pad) using compress, multiply by inner-product weight -----
    print("\n=== Test 2: linear_compressed inner product ===")
    in_dim = 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    w = rng.standard_normal(in_dim) * 0.1
    expected = float(np.dot(x, w))
    print(f"  expected <x,w> = {expected:.6f}")
    print(f"  x = {x.round(4)}")
    print(f"  w = {w.round(4)}")

    x_pad = np.zeros(N); x_pad[:in_dim] = x
    w_pad = np.zeros(N); w_pad[:in_dim] = w

    ct_x = enc_compress(be, x_pad.tolist())
    pt_w = _encode_weight_row_inner_product(be, w_pad)
    out = be._mul_plain_pt(ct_x, pt_w)
    ptd = be._decryptor.decrypt(be._ctx, out)
    decoded = np.array(be._encoder.decode(ptd))
    print(f"  decoded[:8] = {decoded[:8]}")
    print(f"  decoded.mean() = {decoded.mean():.6e}")
    print(f"  decoded.max()  = {decoded.max():.6e}")

    # If coeff[0] holds <x,w>, slot values are FFT of [<x,w>, 0, 0, ...] which is constant <x,w>.
    # So mean should = <x,w>. If mean ~= 0 but some particular slot is large, maybe value is in another coeff.

    # Try interpretation: mean of |decoded|
    print(f"  |decoded| max = {np.abs(decoded).max():.6e}, idx={np.abs(decoded).argmax()}")

    # ----- Test 3: try decode_coeff -----
    print("\n=== Test 3: encoder.decode_coeff (if exists) ===")
    if hasattr(be._encoder, "decode_coeff"):
        coeffs = np.array(be._encoder.decode_coeff(ptd))
        print(f"  coeffs[:8] = {coeffs[:8]}")
        print(f"  coeffs[0]  = {coeffs[0]:.6e}")
        print(f"  |coeffs| max = {np.abs(coeffs).max():.6e}, idx={np.abs(coeffs).argmax()}")
    else:
        print("  decode_coeff not exposed on encoder")
        print("  available:", [m for m in dir(be._encoder) if 'decode' in m or 'encode' in m])


if __name__ == "__main__":
    main()
