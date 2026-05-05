"""Probe: chained no-gather coeff-mode linears at BERT-base width.

Goal: measure cost of one full 768x768 linear in no-gather mode,
across all required splits, and validate correctness.

Steps per linear:
  - For each of S splits (out_per=85, S=10 for 768x768):
    - encode W-poly (build w_poly with out at bit-rev coefs)
    - clone ct, mul_plain, clear rescale, CtoS, set rescale, rescale x2
    - the resulting ct holds split s outputs at slots [bitrev(0..85))
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bitrev(i, bits):
    r = 0
    for k in range(bits):
        r = (r << 1) | ((i >> k) & 1)
    return r


def encode_w_poly_bitrev_out(W, n_in, log_n_half, N, *, input_bitrev=False):
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


def matvec_bitrev_out_split(be, ct, W, n_in, log_n_half, N, *, input_bitrev=False):
    """One no-gather split. Returns slot-encoded ct holding outs at bitrev slots."""
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
    log(f"  num_slots={be._num_slots} max_depth={be._max_depth}")

    log_n_half = (N // 2).bit_length() - 1
    in_dim = 768
    out_dim = 768
    out_per = N // in_dim   # 85
    splits = (out_dim + out_per - 1) // out_per   # 10

    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    expected = W @ x

    # === Single 768x768 linear via 10 splits, NO gather ===
    log(f"\n=== 768x768 single-input no-gather ({splits} splits, out_per={out_per}) ===")

    # Warmup once to register CtoS keys.
    log("Warmup ...")
    ct_x = be.encrypt_coeff(x.tolist())
    W_split = W[:out_per, :]
    _ = matvec_bitrev_out_split(be, ct_x, W_split, in_dim, log_n_half, N)
    log(f"  keys={len(be._registered_shifts)}")

    # Time full linear (10 splits).
    times = []
    out_cts = []
    for rep in range(2):
        ct_x = be.encrypt_coeff(x.tolist())
        out_cts.clear()
        t0 = time.time()
        for s in range(splits):
            lo = s * out_per
            hi = min(lo + out_per, out_dim)
            W_s = W[lo:hi, :]
            out_s = matvec_bitrev_out_split(be, ct_x, W_s, in_dim, log_n_half, N)
            out_cts.append(out_s)
        dt = time.time() - t0
        times.append(dt)
        log(f"  rep{rep}: {dt*1000:.1f}ms ({dt/splits*1000:.1f}ms per split)")

    best = min(times)
    log(f"\n  Best: {best*1000:.1f}ms total, {best/splits*1000:.1f}ms per split")

    # Validate correctness on first split.
    log("\nValidation on split 0:")
    dec = be.decrypt(out_cts[0])
    # Outputs at slot[bitrev(i)] for i in [0, out_per)
    got = np.array([dec[bitrev(i, log_n_half)] for i in range(out_per)])
    err = float(np.max(np.abs(got - expected[:out_per])))
    log(f"  Split 0 err: {err:.3e}  expected[:3]={expected[:3]}  got[:3]={got[:3]}")

    # Validate split 5
    dec = be.decrypt(out_cts[5])
    got = np.array([dec[bitrev(i, log_n_half)] for i in range(out_per)])
    err = float(np.max(np.abs(got - expected[5*out_per:5*out_per + out_per])))
    log(f"  Split 5 err: {err:.3e}")

    # Per-layer projection (single input).
    log(f"\n=== Per-layer projection (single input, no batching) ===")
    one_768x768 = best
    # QKVO: 4 of these
    # W1: 3072x768 → 36 splits of 85
    # W2: 768x3072 → in_dim=3072, out_per = 65536/3072 = 21, splits=37
    qkvo = 4 * one_768x768
    log(f"  QKVO (4 × 768x768):     {qkvo*1000:8.1f} ms")
    # W1 = 36 splits at in=768 → ~same per-split cost
    w1 = 36 * (best/splits)
    log(f"  W1 (3072x768, 36 spl):   {w1*1000:8.1f} ms")
    # W2: in=3072 → encoding cost may differ. Skip for now, estimate.
    w2_est = 37 * (best/splits)
    log(f"  W2 (768x3072 est):       {w2_est*1000:8.1f} ms (rough — different in_dim)")
    log(f"  Linear sum/layer:        {(qkvo+w1+w2_est)*1000:8.1f} ms")
    log(f"  +Non-linear/layer ~50ms  ≈ {(qkvo+w1+w2_est+0.05)*1000:8.1f} ms/layer")
    log(f"  12-layer projection:     {12*(qkvo+w1+w2_est+0.05):8.1f} s (single-input wall clock)")


if __name__ == "__main__":
    main()
