"""Validate NEW coeff_matvec_bitrev_out: output at coeff[bitrev(i, 15)].

If correct, this eliminates the gather entirely from the chain.
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


def encode_w_poly_bitrev_out(W, n_in, log_n_half, N):
    """Build W-poly for output at coeff[bitrev(i, log_n_half)].

    For input x at coeffs [0..n_in), we want
       result[bitrev(i)] = sum_j W[i,j] * x[j]

    => w_poly[(bitrev(i) - j) mod_neg_cyclic 2N] = (sign) * W[i, j].
    """
    m = W.shape[0]
    w_poly = [0.0] * N
    for i in range(m):
        target = bitrev(i, log_n_half)
        for j in range(n_in):
            idx = target - j
            if idx >= 0:
                w_poly[idx] = float(W[i, j])
            else:
                # negacyclic: x^N = -1, so x^(idx) = -x^(idx+N)
                w_poly[idx + N] = -float(W[i, j])
    return w_poly


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
    log_n_half = (N // 2).bit_length() - 1   # 15

    rng = np.random.default_rng(7)
    n_in = 64
    m = 64
    x = rng.standard_normal(n_in) * 0.3
    W = rng.standard_normal((m, n_in)) * 0.05
    expected = W @ x

    # Encode W-poly for bitrev output
    w_poly = encode_w_poly_bitrev_out(W, n_in, log_n_half, N)
    print(f"  w_poly nonzero count: {sum(1 for v in w_poly if v != 0)}")

    # Encrypt x as coefficients [0..n_in)
    ct_x = be.encrypt_coeff(x.tolist())

    # Multiply
    pt_w = be._encode_coeff_pad(w_poly)
    out = be._ops.clone_ct(ct_x)
    while be._ops.depth_of_plaintext(pt_w) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w)
    be._ops.multiply_plain_inplace(out, pt_w)
    # DON'T rescale yet — stay at depth 0 for CtoS
    be._ops.clear_rescale_required(out)

    # Decrypt to verify result is at coeff[bitrev(i)]
    coeffs = be.decrypt_coeff(out)
    # Expected positions
    err = float(np.max(np.abs(np.array([coeffs[bitrev(i, log_n_half)] for i in range(m)]) - expected)))
    print(f"  result at coeff[bitrev(i)]? err={err:.3e}")
    print(f"  expected[:4] = {expected[:4]}")
    print(f"  got at bitrev positions[:4] = {[coeffs[bitrev(i, log_n_half)] for i in range(4)]}")

    if err > 1e-3:
        print("  ❌ encoding wrong; aborting")
        return

    # Now CtoS — should give slot[i] = expected[i]
    cts = be.coeff_to_slot(out)
    for c in cts:
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    ct_slot = cts[0]
    dec = be.decrypt(ct_slot)
    err_slot = float(np.max(np.abs(np.array(dec[:m]) - expected)))
    print(f"  after CtoS: slot[:m] err vs expected = {err_slot:.3e}")
    print(f"  slot[:4] = {dec[:4]}")


if __name__ == "__main__":
    main()
