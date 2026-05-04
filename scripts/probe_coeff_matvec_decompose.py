"""Decompose coeff_matvec_to_slot timing: build w_poly vs encode vs mul vs CtoS."""
import time
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


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
    N = be._num_slots
    print(f"N={N}\n")

    in_dim, out_dim = 128, 128
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    ct_x = be.encrypt_coeff(x.tolist())

    # Warm
    _ = be.coeff_matvec_to_slot(ct_x, W, in_dim=in_dim)

    print("=== Stage decomposition (median of 5) ===\n")
    n_rep = 5

    # 1. Build w_poly (Python list-of-floats of length 2N)
    times = []
    for _ in range(n_rep):
        t = time.time()
        w_poly = [0.0] * be._N
        for i in range(out_dim):
            base = i * in_dim
            for j in range(in_dim):
                w_poly[base + (in_dim - 1 - j)] = float(W[i, j])
        times.append(time.time() - t)
    t_build = np.median(times) * 1000
    print(f"  (1) Python build w_poly        : {t_build:.2f} ms  (len={len(w_poly)})")

    # 2. Encode the poly into a coeff-domain plaintext
    times = []
    for _ in range(n_rep):
        t = time.time()
        _ = be._encode_coeff_pad(w_poly)
        times.append(time.time() - t)
    t_enc = np.median(times) * 1000
    print(f"  (2) encode w_poly (coeff pad)  : {t_enc:.2f} ms")

    # 3. clone_ct + mod_drop_pt + multiply_plain (no rescale)
    pt_w = be._encode_coeff_pad(w_poly)
    times = []
    for _ in range(n_rep):
        t = time.time()
        out = be._ops.clone_ct(ct_x)
        td = be._ops.depth(out)
        pt_w_cp = be._encode_coeff_pad(w_poly)  # need fresh pt each iter (mod_drop is destructive)
        while be._ops.depth_of_plaintext(pt_w_cp) < td:
            be._ops.mod_drop_inplace_pt(pt_w_cp)
        be._ops.multiply_plain_inplace(out, pt_w_cp)
        times.append(time.time() - t)
    t_mul = np.median(times) * 1000
    print(f"  (3) clone+modDrop+mul (no rs)  : {t_mul:.2f} ms")

    # 3b. Same but skip the encode (cached pt)
    times = []
    for _ in range(n_rep):
        # Need fresh pt because mod_drop is destructive AND plaintext can't be cloned cheaply
        t = time.time()
        out = be._ops.clone_ct(ct_x)
        be._ops.multiply_plain_inplace(out, pt_w)  # use same pt — depth already 0
        times.append(time.time() - t)
    t_mul_cached = np.median(times) * 1000
    print(f"  (3b) clone+mul (cached pt)     : {t_mul_cached:.2f} ms")

    # 4. coeff_to_slot alone (C++ heavy lift)
    out = be._ops.clone_ct(ct_x)
    pt_w2 = be._encode_coeff_pad(w_poly)
    be._ops.multiply_plain_inplace(out, pt_w2)
    be._ops.clear_rescale_required(out)

    # Make several copies for repeated CtoS measurement
    times = []
    for _ in range(n_rep):
        out_copy = be._ops.clone_ct(out)
        be._ops.clear_rescale_required(out_copy)
        t = time.time()
        cts = be._ops.coeff_to_slot(out_copy, be._gk)
        times.append(time.time() - t)
    t_ctos = np.median(times) * 1000
    print(f"  (4) coeff_to_slot (C++)        : {t_ctos:.2f} ms  → {len(cts)} cts out")

    # 5. set_rescale_required + rescale on each of 2 cts
    out_copy = be._ops.clone_ct(out)
    be._ops.clear_rescale_required(out_copy)
    cts_orig = be._ops.coeff_to_slot(out_copy, be._gk)
    times = []
    for _ in range(n_rep):
        # need fresh cts each iter (rescale is destructive)
        out_copy = be._ops.clone_ct(out)
        be._ops.clear_rescale_required(out_copy)
        cts2 = be._ops.coeff_to_slot(out_copy, be._gk)
        t = time.time()
        for c in cts2:
            be._ops.set_rescale_required(c)
            be._ops.rescale_inplace(c)
        times.append(time.time() - t)
    t_rs = np.median(times) * 1000
    print(f"  (5) post-CtoS rescale (×2)     : {t_rs:.2f} ms")

    # Full call
    times = []
    for _ in range(n_rep):
        t = time.time()
        _ = be.coeff_matvec_to_slot(ct_x, W, in_dim=in_dim)
        times.append(time.time() - t)
    t_full = np.median(times) * 1000
    print(f"\n  FULL coeff_matvec_to_slot     : {t_full:.2f} ms")

    accounted = t_build + t_enc + t_mul + t_ctos + t_rs
    # subtract t_enc since it's inside t_mul
    accounted = t_build + t_mul + t_ctos + t_rs
    print(f"  Sum of (1)+(3)+(4)+(5)        : {accounted:.2f} ms")
    print(f"  Unaccounted overhead          : {t_full - accounted:.2f} ms")


if __name__ == "__main__":
    main()
