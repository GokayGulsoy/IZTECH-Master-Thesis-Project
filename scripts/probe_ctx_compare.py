"""Isolate: does with-context CtoS at ctos_level=0 give same result as
the no-context (bootstrap-path) CtoS at depth 0?

If YES → with-context code is fine, deep-level encoding is the bug.
If NO  → with-context code itself is broken.
"""
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption import heongpu_bindings as he


def main():
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    scale_boot = be._ops.bootstrapping_scale()

    # Fresh transform context at depth 0 (matches bootstrap-path config)
    ctx0 = he.EncodingTransformContext()
    be._ops.generate_encoding_transform_context(ctx0, scale_boot, 3, 3, 0, -1, True)
    print(f"ctx0: ctos={ctx0.ctos_level()} stoc={ctx0.stoc_level()}")

    rng = np.random.default_rng(7)
    n, h = 64, 64
    x = rng.standard_normal(n) * 0.3
    W = rng.standard_normal((h, n)) * 0.05

    log_n_half = (be._N // 2).bit_length() - 1

    def bitrev(i, bits=log_n_half):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r

    # Build w-poly with bit-rev output encoding
    w = [0.0] * be._N
    for i in range(h):
        target = bitrev(i)
        for j in range(n):
            k = target - j
            if k >= 0:
                w[k] = float(W[i, j])
            else:
                w[k + be._N] = -float(W[i, j])

    pt = be._encode_coeff_pad(w)
    expected = W @ x

    # Method A: no-ctx bootstrap-path CtoS
    ct = be.encrypt_coeff(x.tolist())
    out_a = be._ops.clone_ct(ct)
    be._ops.multiply_plain_inplace(out_a, pt)
    be._ops.clear_rescale_required(out_a)
    cts_a = be._ops.coeff_to_slot(out_a, be._gk)
    for c in cts_a:
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    dec_a = be.decrypt(cts_a[0])
    err_a = float(np.max(np.abs(np.array(dec_a[:h]) - expected)))
    print(f"Method A (no-ctx CtoS): err={err_a:.3e}")

    # Method B: with-ctx CtoS at depth 0
    ct = be.encrypt_coeff(x.tolist())
    out_b = be._ops.clone_ct(ct)
    be._ops.multiply_plain_inplace(out_b, pt)
    be._ops.clear_rescale_required(out_b)
    cts_b = be._ops.coeff_to_slot_ctx(out_b, be._gk, ctx0)
    for c in cts_b:
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    dec_b = be.decrypt(cts_b[0])
    err_b = float(np.max(np.abs(np.array(dec_b[:h]) - expected)))
    print(f"Method B (with-ctx CtoS at d=0): err={err_b:.3e}")

    # Method C: with-ctx CtoS at depth=11 — but must mod_drop input first
    ctx11 = he.EncodingTransformContext()
    be._ops.generate_encoding_transform_context(ctx11, scale_boot, 3, 3, 11, -1, True)

    ct = be.encrypt_coeff(x.tolist())
    out_c = be._ops.clone_ct(ct)
    # Drop ct to depth 11 (this also drops scale appropriately? Or not?)
    while be._ops.depth(out_c) < 11:
        be._ops.mod_drop_inplace_ct(out_c)
    pt_c = be._encode_coeff_pad(w)
    while be._ops.depth_of_plaintext(pt_c) < 11:
        be._ops.mod_drop_inplace_pt(pt_c)
    print(f"  pre-mul C: ct depth={be._ops.depth(out_c)} scale=2^{np.log2(out_c.scale()):.2f}")
    be._ops.multiply_plain_inplace(out_c, pt_c)
    print(f"  post-mul C: depth={be._ops.depth(out_c)} scale=2^{np.log2(out_c.scale()):.2f}")
    be._ops.clear_rescale_required(out_c)
    cts_c = be._ops.coeff_to_slot_ctx(out_c, be._gk, ctx11)
    for c in cts_c:
        print(f"    post-CtoS: depth={be._ops.depth(c)} scale=2^{np.log2(c.scale()):.2f}")
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    dec_c = be.decrypt(cts_c[0])
    err_c = float(np.max(np.abs(np.array(dec_c[:h]) - expected)))
    print(f"Method C (with-ctx CtoS at d=11, fresh enc): err={err_c:.3e}")
    print(f"  got[:4]={dec_c[:4]}")
    print(f"  exp[:4]={expected[:4]}")


if __name__ == "__main__":
    main()
