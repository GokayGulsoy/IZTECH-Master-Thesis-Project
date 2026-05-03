"""Full A2 chain validation:
  L1 (input-contig, output-bitrev) → polyval → StoC → L2 (input-bitrev, output-bitrev)

If correct, this is the closed-loop NEXUS pipeline at ~45ms per linear.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def bitrev(i, bits):
    r = 0
    for k in range(bits):
        r = (r << 1) | ((i >> k) & 1)
    return r


def encode_w_poly_bitrev_out(W, n_in, log_n_half, N, *, input_bitrev=False):
    """W-poly placement.

    If input_bitrev=False: input x at coeffs [0..n_in)
    If input_bitrev=True:  input x at coeffs [bitrev(j) for j in 0..n_in)

    Output always at coeff[bitrev(i, log_n_half)].
    """
    m = W.shape[0]
    w_poly = [0.0] * N
    for i in range(m):
        target = bitrev(i, log_n_half)
        for j in range(n_in):
            src = bitrev(j, log_n_half) if input_bitrev else j
            idx = target - src
            if idx >= 0:
                w_poly[idx] = float(W[i, j])
            else:
                w_poly[idx + N] = -float(W[i, j])
    return w_poly


def matvec_bitrev_out(be, ct, W, n_in, log_n_half, N, input_bitrev=False):
    """Run coeff_matvec with bitrev-out encoding, return slot-encoded ct."""
    w_poly = encode_w_poly_bitrev_out(W, n_in, log_n_half, N, input_bitrev=input_bitrev)
    pt_w = be._encode_coeff_pad(w_poly)
    out = be._ops.clone_ct(ct)
    while be._ops.depth_of_plaintext(pt_w) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w)
    be._ops.multiply_plain_inplace(out, pt_w)
    be._ops.clear_rescale_required(out)
    cts = be.coeff_to_slot(out)
    for c in cts:
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    return cts[0]  # slot[i] = (W·x)[i]


def slot_to_coeff_bitrev(be, ct):
    """slot[i] → coeff[bitrev(i)]. Drops to StoC level + runs StoC."""
    target_lvl = be._ops.slot_to_coeff_level()
    target_depth = 30 - target_lvl
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    return be._ops.slot_to_coeff(ct, zero_ct, be._gk)


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

    N = be._N
    log_n_half = (N // 2).bit_length() - 1

    rng = np.random.default_rng(7)
    n, h, m = 64, 64, 64
    x = rng.standard_normal(n) * 0.3
    W1 = rng.standard_normal((h, n)) * 0.05
    b1 = rng.standard_normal(h) * 0.1
    W2 = rng.standard_normal((m, h)) * 0.05
    b2 = rng.standard_normal(m) * 0.1
    poly_coeffs = [0.0, 0.0, 1.0]   # f(z)=z^2

    z1 = W1 @ x + b1
    a1 = z1 ** 2
    z2 = W2 @ a1 + b2
    print(f"  expected z2[:4] = {z2[:4]}")

    # ── Pipeline ──
    ct = be.encrypt_coeff(x.tolist())   # x at coeffs [0..n)

    t0 = time.time()
    ct = matvec_bitrev_out(be, ct, W1, n, log_n_half, N, input_bitrev=False)
    # Add bias at slot[0..h) (contiguous, after CtoS)
    bias_pad = list(b1) + [0.0] * (be._num_slots - h)
    ct = be.add_plain(ct, bias_pad)
    t_l1 = time.time() - t0
    print(f"  L1 (matvec+CtoS+bias):  {t_l1*1000:.1f}ms  depth={be._ops.depth(ct)}")

    dec = be.decrypt(ct)
    err = float(np.max(np.abs(np.array(dec[:h]) - z1)))
    print(f"    z1 err: {err:.3e}")

    # polyval (slot-domain, layout-agnostic)
    t0 = time.time()
    ct = be.polyval(ct, poly_coeffs)
    t_poly = time.time() - t0
    print(f"  polyval (z^2):          {t_poly*1000:.1f}ms  depth={be._ops.depth(ct)}")

    dec = be.decrypt(ct)
    err = float(np.max(np.abs(np.array(dec[:h]) - a1)))
    print(f"    a1 err: {err:.3e}")

    # StoC: slot[i] → coeff[bitrev(i)]
    t0 = time.time()
    ct = slot_to_coeff_bitrev(be, ct)
    t_stoc = time.time() - t0
    print(f"  StoC:                   {t_stoc*1000:.1f}ms  depth={be._ops.depth(ct)}")

    # Sanity
    coeffs = be.decrypt_coeff(ct)
    err = float(np.max(np.abs(np.array([coeffs[bitrev(j, log_n_half)] for j in range(h)]) - a1)))
    print(f"    a1 at coeff[bitrev(j)]? err={err:.3e}")

    # L2: read from bitrev coeffs, output at bitrev coeffs
    t0 = time.time()
    ct = matvec_bitrev_out(be, ct, W2, h, log_n_half, N, input_bitrev=True)
    bias_pad2 = list(b2) + [0.0] * (be._num_slots - m)
    ct = be.add_plain(ct, bias_pad2)
    t_l2 = time.time() - t0
    print(f"  L2 (matvec+CtoS+bias):  {t_l2*1000:.1f}ms  depth={be._ops.depth(ct)}")

    dec = be.decrypt(ct)
    err = float(np.max(np.abs(np.array(dec[:m]) - z2)))
    print(f"    z2 err: {err:.3e}  ← FINAL")
    print(f"    got[:4] = {dec[:4]}")

    total = t_l1 + t_poly + t_stoc + t_l2
    print(f"\n  TOTAL per-token 2-linear chain: {total*1000:.1f}ms")


if __name__ == "__main__":
    main()
