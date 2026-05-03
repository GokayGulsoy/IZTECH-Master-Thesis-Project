"""SIMD multi-token coefficient packing — chained linear validation.

Pack K tokens, each of dim d_in, into one polynomial:
    coeff[t·n + j] = x_t[j]   for t∈[0,K), j∈[0,d_in)
where n = next_pow2(max(d_in, d_out)) is the stride.

For W-poly with a[i-j] = W[i,j] (negacyclic-wrapped),
mul_plain produces:
    coeff[t·n + i] = (W·x_t)[i]   for all t simultaneously.

This means L1 → L2 → L3 chain in COEFF domain, no CtoS, no gather.
Only CtoS when transitioning to slot-domain non-linear ops.

Validates:
  (a) single-linear correctness for K tokens
  (b) chained two-linear correctness (coeff → mul_plain → mul_plain → CtoS)
  (c) timing
"""
import time
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def next_pow2(x):
    n = 1
    while n < x:
        n <<= 1
    return n


def encode_w_stride(W, stride, N):
    """W-poly: a[i-j mod_neg N] = W[i,j], independent of token index."""
    m, in_dim = W.shape
    w = [0.0] * N
    for i in range(m):
        for j in range(in_dim):
            k = i - j
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
    N = be._N
    print(f"  N={N}")

    # BERT-tiny shape: hidden=128. With stride=128, K=N/128=512 tokens fit.
    # We test with K=8 (BERT-tiny seq) and stride=128.
    rng = np.random.default_rng(7)
    K = 8                # tokens
    d_in = 128
    d_mid = 128          # square so chain works
    d_out = 128
    stride = next_pow2(max(d_in, d_mid, d_out))
    print(f"  K={K}, d_in={d_in}, d_mid={d_mid}, d_out={d_out}, stride={stride}")
    print(f"  cap = N/stride = {N // stride} tokens per ct")

    X = rng.standard_normal((K, d_in)) * 0.3
    W1 = rng.standard_normal((d_mid, d_in)) * 0.05
    W2 = rng.standard_normal((d_out, d_mid)) * 0.05
    Z1 = X @ W1.T          # (K, d_mid) — plaintext expected
    Z2 = Z1 @ W2.T         # (K, d_out)
    print(f"  expected Z1[0,:4] = {Z1[0,:4]}")
    print(f"  expected Z2[0,:4] = {Z2[0,:4]}")

    # ── Pack input ──
    pack = [0.0] * N
    for t in range(K):
        for j in range(d_in):
            pack[t * stride + j] = float(X[t, j])
    ct = be.encrypt_coeff(pack)
    print(f"\n  ct depth={be._ops.depth(ct)} scale=2^{np.log2(ct.scale()):.2f}")

    # ── L1 ──
    w_poly1 = encode_w_stride(W1, stride, N)
    pt_w1 = be._encode_coeff_pad(w_poly1)
    t0 = time.time()
    out = be._ops.clone_ct(ct)
    be._ops.multiply_plain_inplace(out, pt_w1)
    be._ops.rescale_inplace(out)
    t_l1 = time.time() - t0
    print(f"  L1 mul+rescale: {t_l1*1000:.1f}ms  depth={be._ops.depth(out)} scale=2^{np.log2(out.scale()):.2f}")

    coeffs = be.decrypt_coeff(out)
    err1 = max(
        max(abs(coeffs[t * stride + i] - Z1[t, i]) for i in range(d_mid))
        for t in range(K)
    )
    print(f"    L1 multi-token err: {err1:.3e}  (over all K·d_mid={K*d_mid} positions)")

    # ── L2 chained directly in coeff (no CtoS) ──
    w_poly2 = encode_w_stride(W2, stride, N)
    pt_w2 = be._encode_coeff_pad(w_poly2)
    while be._ops.depth_of_plaintext(pt_w2) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w2)
    t0 = time.time()
    be._ops.multiply_plain_inplace(out, pt_w2)
    be._ops.rescale_inplace(out)
    t_l2 = time.time() - t0
    print(f"  L2 mul+rescale: {t_l2*1000:.1f}ms  depth={be._ops.depth(out)} scale=2^{np.log2(out.scale()):.2f}")

    coeffs = be.decrypt_coeff(out)
    err2 = max(
        max(abs(coeffs[t * stride + i] - Z2[t, i]) for i in range(d_out))
        for t in range(K)
    )
    print(f"    L2 multi-token err: {err2:.3e}  (over all K·d_out={K*d_out} positions)")

    # ── L3 chained (test depth budget) ──
    W3 = rng.standard_normal((d_out, d_out)) * 0.05
    Z3 = Z2 @ W3.T
    w_poly3 = encode_w_stride(W3, stride, N)
    pt_w3 = be._encode_coeff_pad(w_poly3)
    while be._ops.depth_of_plaintext(pt_w3) < be._ops.depth(out):
        be._ops.mod_drop_inplace_pt(pt_w3)
    t0 = time.time()
    be._ops.multiply_plain_inplace(out, pt_w3)
    be._ops.rescale_inplace(out)
    t_l3 = time.time() - t0
    coeffs = be.decrypt_coeff(out)
    err3 = max(
        max(abs(coeffs[t * stride + i] - Z3[t, i]) for i in range(d_out))
        for t in range(K)
    )
    print(f"  L3 mul+rescale: {t_l3*1000:.1f}ms  depth={be._ops.depth(out)}  err={err3:.3e}")

    # ── Compare vs production: K calls to nexus_linear ──
    print(f"\n  --- Cost comparison ---")
    print(f"  Multi-token (K={K}): L1+L2+L3 = {(t_l1+t_l2+t_l3)*1000:.1f}ms")
    print(f"  Per-token estimate: nexus_linear ≈ 683ms × {K} × 3 = {683*K*3/1000:.1f}s")
    print(f"  Speedup: {(683*K*3) / ((t_l1+t_l2+t_l3)*1000):.0f}×")

    if err3 < 1e-3:
        print(f"\n  ✅ MULTI-TOKEN CHAIN WORKS — coeff→coeff→coeff at err {err3:.1e}")


if __name__ == "__main__":
    main()
