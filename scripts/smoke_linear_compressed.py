"""Phase 8d-2 smoke + bench: optimized linear via coeff-domain inner product."""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import enc_compress, linear_compressed


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

    rng = np.random.default_rng(0)

    # --- correctness: small M ---
    M_test = 8
    x = rng.normal(size=N).astype(np.float64) * 0.3
    W = rng.normal(size=(M_test, N)).astype(np.float64) * 0.05
    expected = W @ x

    ct_x = enc_compress(be, x)
    t = time.time()
    out = linear_compressed(be, W, ct_x)
    dt = time.time() - t
    print(f"linear_compressed: {dt*1000:.1f}ms for M={M_test}, N={N} ({dt*1000/M_test:.2f} ms/output)")

    max_err = 0.0
    for i in range(M_test):
        pt = be._decryptor.decrypt(be._ctx, out[i])
        decoded = np.array(be._encoder.decode(pt))
        # Output ct holds polynomial p(x) = (W@x)[i] at coeff 0, ~0 elsewhere.
        # decoded.mean() = sum(coeffs)/N = (W@x)[i] / N (constant-poly slot interp).
        # So multiply by N to recover the true value.
        # Wait — actually the constant polynomial p(x) = c has slot evaluation
        # = c at every slot (constant on every root of unity). decoded[k] = c.
        # decoded.mean() = c. So no /N here, unlike the broadcast case which
        # had c spread across only coeff 0 (not constant poly).
        # But we WANT only coeff 0 nonzero — so it's NOT constant poly, it's
        # a single-monomial poly. Then slot-domain decoded[k] = c at every k
        # because... no wait, X^0 IS constant. coeff[0] only nonzero == p(x)=c
        # is THE constant polynomial. So decoded.mean() = c = (W@x)[i].
        got_mean = decoded.mean()
        # Also try directly reading coeff[0] via decode_coeff:
        coeffs = np.array(be._encoder.decode_coeff(pt))
        got_coeff0 = coeffs[0]
        err_mean = abs(got_mean - expected[i])
        err_coeff = abs(got_coeff0 - expected[i])
        max_err = max(max_err, min(err_mean, err_coeff))
        print(f"  i={i}  expect={expected[i]:+.4f}  mean={got_mean:+.4f} (err {err_mean:.2e})  coeff0={got_coeff0:+.4f} (err {err_coeff:.2e})")
    print(f"max err = {max_err:.3e}")
    assert max_err < 1e-2, f"err too large: {max_err}"
    print("CORRECTNESS PASS")

    # --- bench: BERT-base linear (M=768) ---
    print()
    print("=== BERT-base linear bench: 768 outputs, N=4096 input ===")
    M_bench = 768
    W_bench = rng.normal(size=(M_bench, N)).astype(np.float64) * 0.05
    # Warm
    _ = linear_compressed(be, W_bench[:8], ct_x)
    t = time.time()
    out_bench = linear_compressed(be, W_bench, ct_x)
    dt = time.time() - t
    print(f"linear_compressed: {dt:.2f}s for M=768, N=4096 ({dt*1000/M_bench:.2f} ms/output)")
    # Spot-check 4 random outputs
    expected_bench = W_bench @ x
    sample_idx = rng.choice(M_bench, size=4, replace=False)
    for i in sample_idx:
        pt = be._decryptor.decrypt(be._ctx, out_bench[i])
        decoded = np.array(be._encoder.decode(pt))
        err = abs(decoded.mean() - expected_bench[i])
        print(f"  i={i}  expect={expected_bench[i]:+.4f}  got={decoded.mean():+.4f}  err={err:.2e}")


if __name__ == "__main__":
    main()
