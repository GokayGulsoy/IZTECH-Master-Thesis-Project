"""TEST: full 2-linear chain with stride-n input pattern for L2.

Pipeline:
  L1: coeff_matvec (input contig [0..n)) → coeff[(i+1)*n-1]
      → CtoS → slot[bitrev((i+1)*n-1)]
      → polyval (slot, layout-agnostic)
      → StoC → coeff[(i+1)*n-1]
  L2: coeff_matvec_stride (input at coeff[(j+1)*n-1]) → coeff[(i+1)*n-1]
      → CtoS → slot[bitrev((i+1)*n-1)]
      → final read

Requires building W2-poly with: a[(i-j)*n mod 2N] = (sign) * W2[i, j].
"""
import numpy as np
import time

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def encode_w_poly_stride_in(W, n, N):
    """Build W-poly that reads from coeff[(j+1)*n-1] and writes to coeff[(i+1)*n-1].

    Convolution: c_p = Σ_{q+r=p} a_q * b_r.
    Want c_{(i+1)n-1} = Σ_j a_{?} * b_{(j+1)n-1} = Σ_j W[i,j] * b_{(j+1)n-1}.
    So a position = (i+1)n-1 - ((j+1)n-1) = (i-j)*n.
    Negacyclic: a[(i-j)*n + N] = -W[i,j] when (i-j)*n < 0.
    """
    m, in_dim = W.shape
    w = [0.0] * N
    for i in range(m):
        for j in range(in_dim):
            k = (i - j) * n
            if k >= 0:
                w[k] = float(W[i, j])
            else:
                w[k + N] = -float(W[i, j])
    return w


def main():
    print("Init HEonGPU N=2^16, 30-chain...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    print(f"  max_depth={be._max_depth}")

    rng = np.random.default_rng(7)
    n, h, m = 64, 64, 64
    x = rng.standard_normal(n) * 0.3
    W1 = rng.standard_normal((h, n)) * 0.05
    W2 = rng.standard_normal((m, h)) * 0.05
    poly_coeffs = [0.0, 0.0, 1.0]
    z1 = W1 @ x
    a1 = z1 ** 2
    z2 = W2 @ a1
    print(f"  expected z2[:4] = {z2[:4]}")

    # ── L1 ──
    ct = be.encrypt_coeff(x.tolist())
    t0 = time.time()
    cts = be.coeff_matvec_to_slot(ct, W1, in_dim=n)
    ct = cts[0]
    t_l1 = time.time() - t0
    print(f"\n  L1: {t_l1*1000:.1f}ms  depth={be._ops.depth(ct)}")
    targets = be.nexus_target_slots(n, h)

    # polyval — slot, layout-agnostic
    t0 = time.time()
    ct = be.polyval(ct, poly_coeffs)
    t_poly = time.time() - t0
    print(f"  polyval: {t_poly*1000:.1f}ms  depth={be._ops.depth(ct)}")
    dec = be.decrypt(ct)
    err_a1 = float(np.max(np.abs(np.array([dec[targets[j]] for j in range(h)]) - a1)))
    print(f"    a1 at target slots err: {err_a1:.3e}")

    # StoC
    target_lvl = be._ops.slot_to_coeff_level()
    target_depth = be._max_depth - target_lvl
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    t0 = time.time()
    ct = be._ops.slot_to_coeff(ct, zero_ct, be._gk)
    t_stoc = time.time() - t0
    print(f"  StoC: {t_stoc*1000:.1f}ms  depth={be._ops.depth(ct)}, encoding={ct.encoding_type()}")

    coeffs = be.decrypt_coeff(ct)
    err_a1_coeff = float(np.max(np.abs(np.array([coeffs[(j+1)*n-1] for j in range(h)]) - a1)))
    print(f"    a1 at coeff[(j+1)n-1] err: {err_a1_coeff:.3e}")

    # ── L2: stride-input coeff_matvec ──
    w_poly = encode_w_poly_stride_in(W2, n, be._N)
    print(f"  L2 w_poly nonzero count: {sum(1 for v in w_poly if v != 0)}")
    pt_w = be._encode_coeff_pad(w_poly)
    t0 = time.time()
    out = be._ops.clone_ct(ct)
    while be._ops.depth_of_plaintext(pt_w) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w)
    be._ops.multiply_plain_inplace(out, pt_w)
    be._ops.clear_rescale_required(out)
    # CtoS at depth=25 (won't be at 0!) — this is the test
    print(f"  L2 multiply done, depth={be._ops.depth(out)}, attempting CtoS...")
    try:
        cts_l2 = be._ops.coeff_to_slot(out, be._gk)
        for c in cts_l2:
            be._ops.set_rescale_required(c)
            be._ops.rescale_inplace(c)
        ct_l2 = cts_l2[0]
        t_l2 = time.time() - t0
        print(f"  L2: {t_l2*1000:.1f}ms  depth={be._ops.depth(ct_l2)}")
        dec_l2 = be.decrypt(ct_l2)
        targets2 = be.nexus_target_slots(n, m)
        err_z2 = float(np.max(np.abs(np.array([dec_l2[targets2[j]] for j in range(m)]) - z2)))
        print(f"    z2 err at target slots: {err_z2:.3e}  ← FINAL")
        if err_z2 < 1e-3:
            total = t_l1 + t_poly + t_stoc + t_l2
            print(f"\n  ✅ FULL CHAIN WORKS — total {total*1000:.0f}ms per pair of linears (no bootstrap)")
        else:
            # try other read positions
            err_z2_contig = float(np.max(np.abs(np.array(dec_l2[:m]) - z2)))
            print(f"    z2 contig err: {err_z2_contig:.3e}")
            print(f"    dec_l2[targets2[0]]={dec_l2[targets2[0]]:.4f}, expected={z2[0]:.4f}")
    except Exception as e:
        print(f"  L2 CtoS failed: {e}")


if __name__ == "__main__":
    main()
