"""Validate: NO-GATHER nexus chain.

If nexus output stays at bit-rev SLOT positions, polyval is layout-agnostic,
and slot_to_coeff on bit-rev slot data gives CONTIGUOUS coeff[0..m).

Then a 2-linear chain costs:
  nexus_no_gather (~13ms) + polyval (~7ms) + StoC (~4ms) + nexus_no_gather (~13ms)
  = ~37ms per layer, 30× faster than the gather-based version.
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
    n, h, m = 64, 64, 64
    x = rng.standard_normal(n) * 0.3
    W1 = rng.standard_normal((h, n)) * 0.05
    b1 = rng.standard_normal(h) * 0.1
    W2 = rng.standard_normal((m, h)) * 0.05
    b2 = rng.standard_normal(m) * 0.1
    poly_coeffs = [0.0, 0.0, 1.0]   # f(z) = z^2

    z1 = W1 @ x + b1
    a1 = z1 ** 2
    z2 = W2 @ a1 + b2
    print(f"  expected[:4] = {z2[:4]}")

    def bitrev(i, bits=log_n):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r

    # ── ENCRYPTED no-gather chain ──
    ct = be.encrypt_coeff(x.tolist())

    # Linear 1: coeff_matvec_to_slot — outputs at bit-rev slot positions
    t0 = time.time()
    cts = be.coeff_matvec_to_slot(ct, W1, in_dim=n)
    t_l1 = time.time() - t0
    ct = cts[0]
    print(f"  coeff_matvec_to_slot(W1): {t_l1*1000:.1f}ms  depth={be._ops.depth(ct)}")

    targets = be.nexus_target_slots(n, h)
    print(f"  target slots[:8] = {targets[:8]}")

    # Add bias at the actual target positions
    bias_pad = [0.0] * be._num_slots
    for j in range(h):
        bias_pad[targets[j]] = b1[j]
    ct = be.add_plain(ct, bias_pad)

    dec = be.decrypt(ct)
    err_z1 = float(np.max(np.abs(np.array([dec[targets[j]] for j in range(h)]) - z1)))
    print(f"  z1 at target slots? err={err_z1:.3e}")

    # Polyval (element-wise, layout-agnostic)
    t0 = time.time()
    ct = be.polyval(ct, poly_coeffs)
    t_poly = time.time() - t0
    print(f"  polyval: {t_poly*1000:.1f}ms  depth={be._ops.depth(ct)}")

    # StoC: bit-rev slot data → contiguous coeff
    target_lvl = be._ops.slot_to_coeff_level()
    target_depth = 30 - target_lvl
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    t0 = time.time()
    ct_coeff = be._ops.slot_to_coeff(ct, zero_ct, be._gk)
    t_stoc = time.time() - t0
    print(f"  slot_to_coeff: {t_stoc*1000:.1f}ms  depth={be._ops.depth(ct_coeff)}")

    coeffs = be.decrypt_coeff(ct_coeff)
    err_a1_contig = float(np.max(np.abs(np.array(coeffs[:h]) - a1)))
    print(f"  a1 at coeffs[0..h)? err={err_a1_contig:.3e}  ← if ~0, no gather needed!")

    if err_a1_contig > 1e-3:
        # Maybe at bit-rev coeffs?
        err_a1_brev = float(np.max(np.abs(np.array([coeffs[bitrev(j)] for j in range(h)]) - a1)))
        print(f"  a1 at bitrev coeffs? err={err_a1_brev:.3e}")
        # Or maybe scaled? Check magnitude
        print(f"  |coeffs[:8]|={np.abs(coeffs[:8])}")
        print(f"  |a1[:8]|={np.abs(a1[:8])}")
        return

    # Need to bootstrap if depth > 0 to feed next CtoS
    print(f"  pre-bootstrap depth={be._ops.depth(ct_coeff)}")
    # For now skip bootstrap — just attempt linear 2 directly
    # nexus_linear needs depth-0 input, so this will fail. Let's see.
    try:
        t0 = time.time()
        cts2 = be.coeff_matvec_to_slot(ct_coeff, W2, in_dim=h)
        t_l2 = time.time() - t0
        ct2 = cts2[0]
        bias_pad2 = [0.0] * be._num_slots
        for j in range(m):
            bias_pad2[bitrev(j)] = b2[j]
        ct2 = be.add_plain(ct2, bias_pad2)
        dec2 = be.decrypt(ct2)
        z2_got = np.array([dec2[bitrev(j)] for j in range(m)])
        err_z2 = float(np.max(np.abs(z2_got - z2)))
        print(f"  coeff_matvec_to_slot(W2): {t_l2*1000:.1f}ms")
        print(f"  z2 err: {err_z2:.3e}")
        print(f"  got[:4] = {z2_got[:4]}")
    except Exception as e:
        print(f"  Linear 2 failed (likely depth issue): {e}")
        # Bootstrap and retry
        print("  trying bootstrap before linear 2...")
        ct_boot = be.bootstrap(ct_coeff)
        print(f"  post-bootstrap depth={be._ops.depth(ct_boot)}")


if __name__ == "__main__":
    main()
