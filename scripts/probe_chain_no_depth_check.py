"""Try the full nexus chain WITH bootstrap inserted: post-StoC coeff ct
gets bootstrapped back to a usable depth, then a second nexus_linear runs.
"""
import numpy as np
import time

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    # Use 40-chain for headroom
    print("Init HEonGPU N=2^16, 40-chain...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 40,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    print(f"  max_depth={be._max_depth}, CtoS_level={be._ops.coeff_to_slot_level()}, StoC_level={be._ops.slot_to_coeff_level()}")

    rng = np.random.default_rng(7)
    n, h, m = 64, 64, 64
    x = rng.standard_normal(n) * 0.3
    W1 = rng.standard_normal((h, n)) * 0.05
    W2 = rng.standard_normal((m, h)) * 0.05
    z1 = W1 @ x
    z2 = W2 @ z1
    print(f"  expected z2[:4] = {z2[:4]}")

    # ── Step 1: nexus_linear (W1) → slot output at low depth ──
    ct = be.encrypt_coeff(x.tolist())
    t0 = time.time()
    cts = be.coeff_matvec_to_slot(ct, W1, in_dim=n)
    ct = cts[0]
    t_l1 = time.time() - t0
    print(f"\n  L1 (matvec_to_slot): {t_l1*1000:.1f}ms  depth={be._ops.depth(ct)}, encoding={ct.encoding_type()}")
    dec = be.decrypt(ct)
    err1 = float(np.max(np.abs(np.array(dec[:h]) - z1)))
    print(f"    z1 err: {err1:.3e}")

    # ── Step 2: StoC slot → coeff at high depth ──
    target_lvl = be._ops.slot_to_coeff_level()
    target_depth = be._max_depth - target_lvl
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    t0 = time.time()
    ct_coeff = be._ops.slot_to_coeff(ct, zero_ct, be._gk)
    t_stoc = time.time() - t0
    print(f"  StoC: {t_stoc*1000:.1f}ms  depth={be._ops.depth(ct_coeff)}, encoding={ct_coeff.encoding_type()}")
    coeffs = be.decrypt_coeff(ct_coeff)
    log_n = (be._N // 2).bit_length() - 1
    def bitrev(i, bits=log_n):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r
    err_brev = float(np.max(np.abs(np.array([coeffs[bitrev(j)] for j in range(h)]) - z1)))
    print(f"    z1 at coeff[bitrev(j)] err: {err_brev:.3e}")

    # ── Step 3: try CtoS DIRECTLY without bootstrap (test depth-flexibility) ──
    print(f"\n  Test: can CtoS work on depth={be._ops.depth(ct_coeff)} coeff input?")
    try:
        cts_a = be._ops.coeff_to_slot(ct_coeff, be._gk)
        print(f"    ✓ CtoS at depth={be._ops.depth(ct_coeff)} succeeded!")
        print(f"    out depth={be._ops.depth(cts_a[0])}, encoding={cts_a[0].encoding_type()}")
        for c in cts_a:
            be._ops.set_rescale_required(c)
            be._ops.rescale_inplace(c)
        dec_a = be.decrypt(cts_a[0])
        # Slot[i] should hold z1[i]? Or W1·x at bit-rev positions of original
        err_a_contig = float(np.max(np.abs(np.array(dec_a[:h]) - z1)))
        print(f"    slot[:h] err vs z1: {err_a_contig:.3e}")
        # Also try as coefficient-input pattern (since input data was at coeff[bitrev(j)])
    except Exception as e:
        print(f"    ✗ CtoS failed: {e}")
        print(f"    Need bootstrap. Try regular_bootstrapping then CtoS...")
        # Bootstrap path
        ctc = be._clone(ct_coeff)
        while be._ops.depth(ctc) < be._max_depth:
            be._ops.mod_drop_inplace_ct(ctc)
        ct_b = be._ops.regular_bootstrapping(ctc, be._gk, be._rk)
        print(f"    boot: depth={be._ops.depth(ct_b)}, encoding={ct_b.encoding_type()}")
        # now what?


if __name__ == "__main__":
    main()
