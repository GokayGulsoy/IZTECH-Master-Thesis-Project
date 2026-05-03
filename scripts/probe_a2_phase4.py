"""NEXUS Phase 4 probe — chained linears via multi-level CtoS/StoC contexts.

Pipeline (with bit-rev-output coeff_matvec, no gather):
  ct_coeff(d=0)
    \u2192 mul_plain(W1)         depth=0   (no rescale)
    \u2192 CtoS(ctx_A)            depth\u22480+pieces \u2248 4
    \u2192 polyval(z\u00b2)            depth\u22486
    \u2192 StoC(ctx_A)            depth\u22489
    \u2192 mul_plain(W2)         depth=9   (no rescale)
    \u2192 CtoS(ctx_B)            depth\u224813
    \u2192 (decrypt; check vs expected z2)

Validates that CtoS works at depth>0 with a freshly precomputed transform context.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption import heongpu_bindings as he


def bitrev(i, bits):
    r = 0
    for k in range(bits):
        r = (r << 1) | ((i >> k) & 1)
    return r


def encode_w_bitrev_out(W, in_dim, log_n_half, N, *, input_bitrev=False):
    m = W.shape[0]
    w = [0.0] * N
    for i in range(m):
        target = bitrev(i, log_n_half)
        for j in range(in_dim):
            src = bitrev(j, log_n_half) if input_bitrev else j
            k = target - src
            if k >= 0:
                w[k] = float(W[i, j])
            else:
                w[k + N] = -float(W[i, j])
    return w


def main():
    print("Init HEonGPU N=2^16, 30-chain (Phase 3 config)...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()  # also gens Galois key from bootstrap key indexes
    print(f"  max_depth={be._max_depth}, Q_size={be._max_depth + 1}")

    # ---- precompute two transform contexts ----
    scale_boot = be._ops.bootstrapping_scale()
    print(f"  scale_boot = 2^{np.log2(scale_boot):.1f}")

    ctx_A = he.EncodingTransformContext()
    # ctos at depth 0 (post-mul, no rescale)
    # stoc at depth 7 (post-polyval z²: CtoS adds ~5, polyval adds 2)
    be._ops.generate_encoding_transform_context(
        ctx_A, scale_boot, 3, 3, 0, 7, True
    )
    print(f"  ctx_A: ctos={ctx_A.ctos_level()}  stoc={ctx_A.stoc_level()}  "
          f"piece=({ctx_A.ctos_piece()},{ctx_A.stoc_piece()})")
    print(f"  ctx_A keys (first 8): {ctx_A.key_indexs()[:8]}  total={len(ctx_A.key_indexs())}")

    # ctx_B: CtoS for L2, after StoC of ctx_A. Empirical depth post-StoC TBD.
    # If ctx_A stoc=7 and StoC consumes 4 levels, post-StoC ≈ depth 11.
    ctos_b_candidate = 11
    print(f"  trying ctx_B ctos_start = {ctos_b_candidate}")
    ctx_B = he.EncodingTransformContext()
    be._ops.generate_encoding_transform_context(
        ctx_B, scale_boot, 3, 3, ctos_b_candidate, -1, True
    )
    print(f"  ctx_B: ctos={ctx_B.ctos_level()}  stoc={ctx_B.stoc_level()}")

    # Sanity: are key_indexs the same? If yes, our existing GK covers both.
    same_keys = set(ctx_A.key_indexs()) == set(ctx_B.key_indexs())
    print(f"  ctx_A.keys == ctx_B.keys ? {same_keys}")
    if not same_keys:
        missing = set(ctx_B.key_indexs()) - set(ctx_A.key_indexs())
        print(f"  missing keys for ctx_B: {len(missing)} ({sorted(missing)[:8]})")
        # Need to regenerate Galois key with union; for now check overlap with bootstrap keys
        boot_keys = set(be._ops.bootstrapping_key_indexs())
        missing_from_boot = set(ctx_B.key_indexs()) - boot_keys
        print(f"  missing keys for ctx_B from bootstrap-key set: {len(missing_from_boot)}")
        if missing_from_boot:
            # regenerate gk with union
            union = sorted(boot_keys | set(ctx_A.key_indexs()) | set(ctx_B.key_indexs()))
            print(f"  regenerating gk for union of {len(union)} shifts")
            kg = be._hg.KeyGenerator(be._ctx)
            be._gk = kg.generate_galois_key(be._ctx, be._sk, union)

    # ---- Pipeline test ----
    rng = np.random.default_rng(7)
    n, h, m = 64, 64, 64
    log_n_half = (be._N // 2).bit_length() - 1
    x = rng.standard_normal(n) * 0.3
    W1 = rng.standard_normal((h, n)) * 0.05
    W2 = rng.standard_normal((m, h)) * 0.05
    poly_coeffs = [0.0, 0.0, 1.0]
    z1 = W1 @ x
    a1 = z1 ** 2
    z2 = W2 @ a1
    print(f"\n  expected z2[:4] = {z2[:4]}")

    # L1 — input contig, output bit-rev (use ctx_A)
    ct = be.encrypt_coeff(x.tolist())
    w_poly1 = encode_w_bitrev_out(W1, n, log_n_half, be._N, input_bitrev=False)
    pt_w1 = be._encode_coeff_pad(w_poly1)

    t0 = time.time()
    out = be._ops.clone_ct(ct)
    while be._ops.depth_of_plaintext(pt_w1) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w1)
    be._ops.multiply_plain_inplace(out, pt_w1)
    be._ops.clear_rescale_required(out)
    cts_a = be._ops.coeff_to_slot_ctx(out, be._gk, ctx_A)
    # rescale to align scale post-CtoS
    for c in cts_a:
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    ct = cts_a[0]
    t_l1 = time.time() - t0
    print(f"  L1 mul+CtoS:  {t_l1*1000:.1f}ms  depth={be._ops.depth(ct)}")
    dec = be.decrypt(ct)
    err1 = float(np.max(np.abs(np.array(dec[:h]) - z1)))
    print(f"    z1 err: {err1:.3e}")
    if err1 > 1e-3:
        print("  ABORT: L1 broken")
        return

    # polyval (slot, layout-agnostic)
    t0 = time.time()
    ct = be.polyval(ct, poly_coeffs)
    t_poly = time.time() - t0
    print(f"  polyval:      {t_poly*1000:.1f}ms  depth={be._ops.depth(ct)}")

    # StoC via ctx_A
    target_depth = ctx_A.stoc_level()
    while be._ops.depth(ct) < target_depth:
        be._ops.mod_drop_inplace_ct(ct)
    zero_ct = be.encrypt([0.0] * be._num_slots)
    while be._ops.depth(zero_ct) < target_depth:
        be._ops.mod_drop_inplace_ct(zero_ct)
    t0 = time.time()
    ct = be._ops.slot_to_coeff_ctx(ct, zero_ct, be._gk, ctx_A)
    t_stoc = time.time() - t0
    print(f"  StoC(ctx_A):  {t_stoc*1000:.1f}ms  depth={be._ops.depth(ct)}, encoding={ct.encoding_type()}")
    coeffs = be.decrypt_coeff(ct)
    err_a1 = float(np.max(np.abs(np.array([coeffs[bitrev(j, log_n_half)] for j in range(h)]) - a1)))
    print(f"    a1 at coeff[bitrev(j)] err: {err_a1:.3e}")
    if err_a1 > 1e-3:
        print("  ABORT: StoC broken")
        return

    # Now check StoC output depth matches ctx_B's expected ctos_level
    actual = be._ops.depth(ct)
    expected = ctx_B.ctos_level()
    print(f"  depth post-StoC = {actual}, ctx_B expects {expected}")

    if actual != expected:
        # adjust by mod-drops
        if actual < expected:
            while be._ops.depth(ct) < expected:
                be._ops.mod_drop_inplace_ct(ct)
        else:
            print(f"  WARNING: StoC produces depth {actual} > ctx_B.ctos_level={expected}; need different ctx_B")

    # L2 — input bit-rev (since coeffs are at bitrev positions), output bit-rev
    w_poly2 = encode_w_bitrev_out(W2, h, log_n_half, be._N, input_bitrev=True)
    pt_w2 = be._encode_coeff_pad(w_poly2)
    t0 = time.time()
    out = be._ops.clone_ct(ct)
    while be._ops.depth_of_plaintext(pt_w2) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w2)
    be._ops.multiply_plain_inplace(out, pt_w2)
    # Try variant A: rescale BEFORE CtoS (consumes 1 level, but aligns scale)
    be._ops.rescale_inplace(out)
    print(f"  L2 mul+rescale done, depth={be._ops.depth(out)}, attempting CtoS(ctx_B)...")
    # Need to update ctx_B to expect this depth
    ctx_B2 = he.EncodingTransformContext()
    be._ops.generate_encoding_transform_context(
        ctx_B2, scale_boot, 3, 3, be._ops.depth(out), -1, True
    )
    print(f"  ctx_B2: ctos={ctx_B2.ctos_level()}  stoc={ctx_B2.stoc_level()}")
    try:
        cts_b = be._ops.coeff_to_slot_ctx(out, be._gk, ctx_B2)
        for c in cts_b:
            be._ops.set_rescale_required(c)
            be._ops.rescale_inplace(c)
        ct2 = cts_b[0]
        t_l2 = time.time() - t0
        print(f"  L2 CtoS:      {t_l2*1000:.1f}ms  depth={be._ops.depth(ct2)}")
        dec2 = be.decrypt(ct2)
        err2 = float(np.max(np.abs(np.array(dec2[:m]) - z2)))
        print(f"    z2 err: {err2:.3e}  ← FINAL")
        if err2 < 1e-3:
            total = t_l1 + t_poly + t_stoc + t_l2
            print(f"\n  ✅ FULL 2-LINEAR CHAIN: {total*1000:.0f}ms (NO BOOTSTRAP)")
            print(f"     vs production matmul_plain @ ~1283ms × 2 = 2566ms → {2566/(total*1000):.1f}× speedup")
        else:
            print(f"  err too high; got[:4]={dec2[:4]}")
    except Exception as e:
        print(f"  L2 CtoS FAILED: {e}")


if __name__ == "__main__":
    main()
