"""Phase 7e-4 smoke: column-major LayerNorm parity.

Validates layernorm_colmajor against torch LayerNorm reference using
the same polynomial inverse-sqrt approximation.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor, layernorm_colmajor,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # BERT-base: hidden=768 (non-pow2 ⇒ internally padded to 1024).
    L, hidden = 32, 768
    log(f"Config L={L} hidden={hidden}")

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 14,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth}")

    rng = np.random.default_rng(13)
    X = rng.standard_normal((L, hidden)) * 0.5
    gamma = rng.standard_normal(hidden) * 0.1 + 1.0
    beta = rng.standard_normal(hidden) * 0.05

    # Polynomial inverse-sqrt over interval centred on typical variance.
    # We pick a very simple polynomial so the parity test is meaningful:
    # invsqrt(x) ≈ p(x) on x ∈ [a, b] using degree-3 minimax.
    # For the smoke test we use a tight interval near σ²≈1.
    a, b = -1.0, 1.0  # trivial interval so absorb is identity
    # Fit polynomial on actual variance domain.
    xs = np.linspace(0.1, 5.0, 200)
    ys = 1.0 / np.sqrt(xs)
    coeffs = np.polyfit(xs, ys, 3)[::-1].tolist()  # power basis [c0..c3]

    def py_invsqrt_poly(v):
        return sum(c * (v ** k) for k, c in enumerate(coeffs))

    # Reference LayerNorm computed with the SAME poly approximation
    # (so we measure FHE precision, not poly approximation error).
    mu = X.mean(axis=1, keepdims=True)
    centred = X - mu
    var = (centred ** 2).mean(axis=1, keepdims=True)
    inv_sigma = np.array([[py_invsqrt_poly(v[0])] for v in var])
    Y_ref = gamma * centred * inv_sigma + beta
    log(f"  Y ref shape {Y_ref.shape} range [{Y_ref.min():.3f}, {Y_ref.max():.3f}]")

    log("Pack X col-major...")
    X_ct = pack_colmajor(be, X, L=L, head_dim=hidden)

    log("LayerNorm col-major...")
    t = time.time()
    Y_ct = layernorm_colmajor(
        be, X_ct, L=L, hidden_dim=hidden,
        invsqrt_power_coeffs=coeffs,
        invsqrt_interval=(a, b),
        gamma=gamma, beta=beta,
    )
    log(f"  wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(Y_ct)}")

    Y_got = unpack_colmajor(be, Y_ct, L=L, head_dim=hidden)
    err = np.max(np.abs(Y_got - Y_ref))
    log(f"  max err: {err:.3e}")
    if err > 1e-3:
        log("FAIL")
        log(f"  got[0,:5]: {Y_got[0, :5]}")
        log(f"  ref[0,:5]: {Y_ref[0, :5]}")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
