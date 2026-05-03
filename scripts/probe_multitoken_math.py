"""Check: does the stride encoding work for K=1?  K=2?"""
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
    N = be._N
    rng = np.random.default_rng(7)

    d, stride = 8, 16
    W = rng.standard_normal((d, d)) * 0.05
    print(f"W diagonals (testing if Toeplitz?):")
    for diag_idx in [-2, -1, 0, 1, 2]:
        diag = [W[i, i-diag_idx] for i in range(d) if 0 <= i-diag_idx < d]
        print(f"  diag {diag_idx:+d}: {[f'{v:.3f}' for v in diag]}")

    # Plaintext check: build w_poly with current encoding, do convolution
    w = [0.0] * N
    for i in range(d):
        for j in range(d):
            k = i - j
            if k >= 0:
                w[k] = float(W[i, j])
            else:
                w[k + N] = -float(W[i, j])

    # K=1, single token
    x = rng.standard_normal(d) * 0.3
    pack = [0.0] * N
    for j in range(d):
        pack[j] = float(x[j])

    # Negacyclic convolution in plaintext: c[p] = Σ_{q,r: q+r=p mod N} a[q]·b[r] - sign
    # For small d we just brute-force the sum over q in 0..N-1 (most are zero)
    def neg_cyc_conv(a, b, N):
        c = [0.0] * N
        for q in range(N):
            if a[q] == 0:
                continue
            for r in range(N):
                if b[r] == 0:
                    continue
                p = (q + r) % N
                sign = -1 if (q + r) >= N else 1
                c[p] += sign * a[q] * b[r]
        return c

    c = neg_cyc_conv(w, pack, N)
    expected_y = W @ x
    err1 = max(abs(c[i] - expected_y[i]) for i in range(d))
    print(f"\nK=1 single-token: err = {err1:.3e}  (should be ≈0)")

    # K=2, two tokens
    x2 = rng.standard_normal(d) * 0.3
    pack2 = [0.0] * N
    for j in range(d):
        pack2[j] = float(x[j])
        pack2[stride + j] = float(x2[j])
    c2 = neg_cyc_conv(w, pack2, N)
    y0 = W @ x
    y1 = W @ x2
    err_t0 = max(abs(c2[i] - y0[i]) for i in range(d))
    err_t1 = max(abs(c2[stride + i] - y1[i]) for i in range(d))
    print(f"K=2 token0: err at coeff[0..d) = {err_t0:.3e}")
    print(f"K=2 token1: err at coeff[stride..stride+d) = {err_t1:.3e}")

    # What's in c2[d..stride)?  (gap between tokens)
    print(f"\nc2 in gap [d..stride):")
    print(f"  values: {[f'{c2[i]:.4f}' for i in range(d, stride)]}")
    print(f"\nGap values are CROSS-TOKEN cross-talk from convolution.")


if __name__ == "__main__":
    main()
