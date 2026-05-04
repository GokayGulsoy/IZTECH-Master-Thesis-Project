"""Phase 8d smoke: tiny matmul via compress + decompress + matrix_mul."""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import (
    enc_compress, decompress, matrix_mul, required_galois_elts,
)


def main() -> None:
    N_LOG = 12
    N = 1 << N_LOG

    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60, 50, 50, 50),
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        sec_none=True,
    )
    print(f"Backend ready: N={be._N}")

    elts = required_galois_elts(N)
    gk = be._kg.generate_galois_key_elts(be._ctx, be._sk, elts)

    rng = np.random.default_rng(42)
    # Use small M (output dim) so smoke runs fast: M=8.
    M = 8
    x = rng.normal(size=N).astype(np.float64) * 0.3
    W = rng.normal(size=(M, N)).astype(np.float64) * 0.1

    expected = W @ x
    print(f"expected[:M] = {expected.round(4)}")

    ct = enc_compress(be, x)
    t = time.time()
    dec = decompress(be, ct, gk)
    print(f"decompress: {time.time()-t:.2f}s for {len(dec)} cts")

    t = time.time()
    out_cts = matrix_mul(be, W, dec)
    print(f"matrix_mul: {time.time()-t:.2f}s for M={M}, N={N} (={M*N} mul_plain)")

    max_err = 0.0
    for i in range(M):
        pt = be._decryptor.decrypt(be._ctx, out_cts[i])
        decoded = np.array(be._encoder.decode(pt))
        # decoded is the polynomial coeffs; constant value = mean over N coeffs
        # = sum / N, but only coeff 0 is nonzero (~ N * true value, then * 1/N
        # by our pre-scaling => coeff[0] ≈ true value). decoded.mean() picks it.
        # Actually decoded.mean() = decoded.sum()/N = (N*true)/N = true. Good.
        got = decoded.mean()
        err = abs(got - expected[i])
        max_err = max(max_err, err)
        print(f"  i={i}  expect={expected[i]:+.4f}  got={got:+.4f}  err={err:.2e}")

    print(f"max err = {max_err:.3e}")
    assert max_err < 1e-2, f"matrix_mul err too large: {max_err}"
    print("PASS")


if __name__ == "__main__":
    main()
