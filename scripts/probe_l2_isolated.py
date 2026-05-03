"""Isolate L2 encoding correctness.

Encrypt a coefficient-encoded ct with values at coeff[bitrev(j,15)] = a1[j],
mimicking StoC output. Apply L2 W2-poly multiplication, CtoS at depth 0.
Compare vs expected z2 = W2 @ a1.
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

    rng = np.random.default_rng(7)
    h, m = 64, 64
    a1 = rng.standard_normal(h) * 0.1
    W2 = rng.standard_normal((m, h)) * 0.05
    z2 = W2 @ a1
    log_n_half = (be._N // 2).bit_length() - 1

    def bitrev(i, bits=log_n_half):
        r = 0
        for k in range(bits):
            r = (r << 1) | ((i >> k) & 1)
        return r

    # Build poly: data at coeff[bitrev(j,15)] = a1[j], rest = 0
    data = [0.0] * be._N
    for j in range(h):
        data[bitrev(j)] = float(a1[j])
    ct = be.encrypt_coeff(data)
    print(f"ct depth={be._ops.depth(ct)} scale=2^{np.log2(ct.scale()):.2f} encoding={ct.encoding_type()}")

    # Verify decrypt_coeff round trip
    coeffs = be.decrypt_coeff(ct)
    err_rt = float(np.max(np.abs(np.array([coeffs[bitrev(j)] for j in range(h)]) - a1)))
    print(f"  round-trip a1 err: {err_rt:.3e}")

    # Build W2-poly with input_bitrev=True, output bitrev
    w = [0.0] * be._N
    for i in range(m):
        target = bitrev(i)
        for j in range(h):
            src = bitrev(j)
            k = target - src
            if k >= 0:
                w[k] = float(W2[i, j])
            else:
                w[k + be._N] = -float(W2[i, j])
    pt_w = be._encode_coeff_pad(w)
    out = be._ops.clone_ct(ct)
    be._ops.multiply_plain_inplace(out, pt_w)
    be._ops.clear_rescale_required(out)
    print(f"  post-mul: depth={be._ops.depth(out)} scale=2^{np.log2(out.scale()):.2f}")

    # CtoS at depth 0
    cts = be._ops.coeff_to_slot(out, be._gk)
    for c in cts:
        be._ops.set_rescale_required(c)
        be._ops.rescale_inplace(c)
    dec = be.decrypt(cts[0])
    err = float(np.max(np.abs(np.array(dec[:m]) - z2)))
    print(f"  z2 err: {err:.3e}")
    print(f"    got[:4]={dec[:4]}")
    print(f"    exp[:4]={z2[:4]}")


if __name__ == "__main__":
    main()
