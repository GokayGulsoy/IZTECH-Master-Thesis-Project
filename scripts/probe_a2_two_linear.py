"""Validate A2: two consecutive nexus_linears with element-wise non-linear in between.

Chain:
  x (coeff) → nexus_linear(W1) → bit-rev slot
            → polyval (relu^2 as proxy for GELU/LN's polynomial pieces)
            → slot_to_coeff → contiguous coeff
            → nexus_linear(W2) → bit-rev slot
            → decode

If output matches W2 · poly(W1 · x + b1) + b2, then the A2 chain works
end-to-end and we can commit to the rewrite.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16 (bootstrap chain)...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    log_n = (be._num_slots).bit_length() - 1

    rng = np.random.default_rng(7)
    n = 64                 # input dim
    h = 64                 # hidden dim (after W1)
    m = 64                 # output dim
    x = rng.standard_normal(n) * 0.3
    W1 = rng.standard_normal((h, n)) * 0.05
    b1 = rng.standard_normal(h) * 0.1
    W2 = rng.standard_normal((m, h)) * 0.05
    b2 = rng.standard_normal(m) * 0.1
    poly_coeffs = [0.0, 0.0, 1.0]   # f(z) = z^2 (depth-1 poly)

    # plaintext expected
    z1 = W1 @ x + b1                                 # (h,)
    a1 = np.polyval(poly_coeffs[::-1], z1)           # element-wise z^2
    z2 = W2 @ a1 + b2                                # (m,)

    print(f"  expected[:4] = {z2[:4]}")

    # ── Encrypted ──
    t0_total = time.time()
    ct = be.encrypt_coeff(x.tolist())                       # coeff input
    t0 = time.time(); ct = be.nexus_linear(ct, W1, in_dim=n, bias=b1.tolist()); t_l1 = time.time()-t0
    print(f"  nexus_linear(W1): {t_l1*1000:.1f}ms  depth={be._ops.depth(ct)}")

    # Sanity: decrypt slots and check W1·x+b1 sits at bit-rev positions [bitrev(0)..bitrev(h-1))
    dec = be.decrypt(ct)
    def bitrev(i, bits=log_n):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r
    # nexus_linear places result at slot [0..m) (its gather puts it there)
    err_l1 = float(np.max(np.abs(np.array(dec[:h]) - z1)))
    print(f"  W1 output err: {err_l1:.3e}  (slots [0..h))")

    # Element-wise polynomial (operates on slot encoding)
    t0 = time.time(); ct = be.polyval(ct, poly_coeffs); t_poly = time.time()-t0
    print(f"  polyval (z^2):    {t_poly*1000:.1f}ms  depth={be._ops.depth(ct)}")

    dec = be.decrypt(ct)
    err_poly = float(np.max(np.abs(np.array(dec[:h]) - a1)))
    print(f"  poly output err: {err_poly:.3e}  (slots [0..h))")

    # We need to send a1 to next linear's coeff input. Currently a1 is at
    # contiguous slots [0..h). slot_to_coeff would put it at bit-rev coeffs.
    # Need to PERMUTE first OR reverse the gather inside nexus_linear so
    # that the polyval data ends up at bit-rev SLOT positions.

    # Approach: feed the polyval output directly to slot_to_coeff with no
    # extra permute. See where the values land.
    target_lvl = be._ops.slot_to_coeff_level()
    target_depth = 30 - target_lvl
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    t0 = time.time(); ct_coeff = be._ops.slot_to_coeff(ct, zero_ct, be._gk); t_stoc = time.time()-t0
    print(f"  slot_to_coeff:    {t_stoc*1000:.1f}ms  depth={be._ops.depth(ct_coeff)}")

    coeffs_dec = be.decrypt_coeff(ct_coeff)
    # Check if a1 is at coeffs [0..h) or at bitrev positions
    err_at_contig = float(np.max(np.abs(np.array(coeffs_dec[:h]) - a1)))
    print(f"  a1 at coeffs[0..h)?   err={err_at_contig:.3e}")
    err_at_bitrev = float(np.max(np.abs(np.array([coeffs_dec[bitrev(j)] for j in range(h)]) - a1)))
    print(f"  a1 at coeffs[bitrev(j)]? err={err_at_bitrev:.3e}")


if __name__ == "__main__":
    main()
