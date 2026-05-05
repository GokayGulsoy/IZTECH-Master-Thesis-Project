"""Probe: no-gather coeff-mode linear at BERT-base width with FIXED bit-rev.

Fixes from v1:
- out_per must be a power of 2 (bit-rev block size). Use out_per=64.
- Vectorize w_poly construction in numpy (was ~100ms Python).

For 768x768: 12 splits of 64.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bitrev_array(M, log_M):
    """Vectorized bit-reversal of integers [0, M)."""
    a = np.arange(M, dtype=np.int64)
    r = np.zeros(M, dtype=np.int64)
    for k in range(log_M):
        r = (r << 1) | ((a >> k) & 1)
    return r


def build_w_poly_bitrev_out(W, log_n_half, N, *, input_bitrev=False):
    """Vectorized W-poly construction.

    pt_W[idx_i_j] = W[i, j]   if idx_i_j >= 0
    pt_W[idx_i_j + N] = -W[i, j]   else
    where idx_i_j = bitrev(i, log_n_half) - (bitrev(j, log_n_half) if input_bitrev else j)

    Output positions cycle through bitrev positions modulo N (anti-cyclic = sign flip).
    """
    m, n = W.shape
    bi = bitrev_array(m, log_n_half)        # shape (m,)
    if input_bitrev:
        bj = bitrev_array(n, log_n_half)    # shape (n,)
    else:
        bj = np.arange(n, dtype=np.int64)

    # idx[i,j] = bi[i] - bj[j]
    idx = bi[:, None] - bj[None, :]   # shape (m, n)
    sign = np.where(idx >= 0, 1.0, -1.0)
    pos = np.where(idx >= 0, idx, idx + N)

    w_poly = np.zeros(N, dtype=np.float64)
    # Some (i,j) pairs may collide on the same coef (in general, they do not for valid m,n).
    # We accumulate via add.at for safety (handles dupes correctly).
    np.add.at(w_poly, pos, sign * W)
    return w_poly.tolist()


def matvec_bitrev_out_split(be, ct, W, log_n_half, N, *, input_bitrev=False):
    w_poly = build_w_poly_bitrev_out(W, log_n_half, N, input_bitrev=input_bitrev)
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
    return cts[0]


def main():
    N = 1 << 16
    log(f"Init N={N}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()

    log_n_half = (N // 2).bit_length() - 1   # 15
    in_dim = 768
    out_dim = 768
    out_per = 64                       # power of 2 for bit-rev
    splits = (out_dim + out_per - 1) // out_per   # 12

    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    expected = W @ x
    log(f"in={in_dim} out={out_dim} out_per={out_per} splits={splits}")

    log("Warmup ...")
    ct_x = be.encrypt_coeff(x.tolist())
    W_split = W[:out_per, :]
    out_warm = matvec_bitrev_out_split(be, ct_x, W_split, log_n_half, N)
    log(f"  keys={len(be._registered_shifts)}")

    # Validate split 0 correctness.
    dec = be.decrypt(out_warm)
    bi64 = bitrev_array(out_per, 6)   # bit-rev within out_per=64
    # Wait — that's WRONG. bit-rev is of i in [0, out_per) with log_n_half bits, not log out_per.
    # Outputs lie at bitrev(i, log_n_half) for i in [0, out_per).
    bi_full = bitrev_array(out_per, log_n_half)
    got = np.array([dec[bi_full[i]] for i in range(out_per)])
    err = float(np.max(np.abs(got - expected[:out_per])))
    log(f"  Split 0 err: {err:.3e}  exp[:3]={expected[:3]} got[:3]={got[:3]}")

    if err > 1e-2:
        log("  BAD CORRECTNESS — abort timing")
        return

    log("\nTiming full 768x768 (12 splits, encode + mul + CtoS no gather):")
    times = []
    for rep in range(2):
        ct_x = be.encrypt_coeff(x.tolist())
        t0 = time.time()
        outs = []
        for s in range(splits):
            lo = s * out_per
            hi = min(lo + out_per, out_dim)
            outs.append(matvec_bitrev_out_split(be, ct_x, W[lo:hi, :], log_n_half, N))
        dt = time.time() - t0
        times.append(dt)
        log(f"  rep{rep}: {dt*1000:.1f}ms  ({dt/splits*1000:.1f}ms per split)")

    best = min(times)

    # Validate full output by stitching splits.
    full = np.zeros(out_dim)
    for s, ct in enumerate(outs):
        dec = be.decrypt(ct)
        for i in range(out_per):
            if s * out_per + i < out_dim:
                full[s * out_per + i] = dec[bi_full[i]]
    err_full = float(np.max(np.abs(full - expected)))
    log(f"\n  Full 768x768 err: {err_full:.3e}")

    log(f"\n  Per 768x768: {best*1000:.1f}ms")
    log(f"  Per layer (4 QKVO + W1[36spl] + W2[~48spl]):")
    per_split = best/splits
    qkvo = 4 * best
    w1 = 36 * per_split    # 3072/64 splits, in_dim=768 same per-split
    # W2: in_dim=3072, so out_per_W2 = N // 3072 rounded down to power of 2 = 16
    # splits = 768/16 = 48
    w2 = 48 * per_split
    log(f"    QKVO {qkvo*1000:.1f}ms  W1 {w1*1000:.1f}ms  W2~{w2*1000:.1f}ms")
    log(f"    Linear/layer ≈ {(qkvo+w1+w2)*1000:.1f}ms")
    log(f"    12-layer linear: ~{12*(qkvo+w1+w2):.1f}s")


if __name__ == "__main__":
    main()
