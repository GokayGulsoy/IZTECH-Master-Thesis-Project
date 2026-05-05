"""Phase 8k smoke: verify multi-ct col-major primitives at L=128, N=2^16.

Tests pack/unpack/add/LN/linear correctness against numpy reference, all
in <30s. Gates the L=128 e2e bench.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor_multi, unpack_colmajor_multi,
    add_multi, sub_multi,
    layernorm_colmajor_multi,
    linear_colmajor_multi_streaming,
    prepare_colmajor_keys,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 128
    hidden = 768
    N = 1 << 16
    log(f"Init backend N={N} chain=24...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 24,
        p_prime_bits=(60, 60),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be._num_slots}")

    log("Register BSGS keys (max_dim=hidden=768)...")
    n_new = prepare_colmajor_keys(be, L=L, max_dim=hidden)
    log(f"  +{n_new} shifts")

    rng = np.random.default_rng(0)
    X = rng.standard_normal((L, hidden)) * 0.1

    # 1. PACK / UNPACK roundtrip
    log("=== test pack/unpack roundtrip ===")
    cts = pack_colmajor_multi(be, X, L=L, hidden=hidden)
    log(f"  packed into {len(cts)} cts (expected 3 = 768/256)")
    X_back = unpack_colmajor_multi(be, cts, L=L, hidden=hidden)
    err = np.max(np.abs(X_back - X))
    log(f"  max round-trip err: {err:.3e}")
    assert err < 1e-5, f"pack/unpack err too high: {err}"

    # 2. ADD test
    log("=== test add_multi ===")
    Y = rng.standard_normal((L, hidden)) * 0.1
    y_cts = pack_colmajor_multi(be, Y, L=L, hidden=hidden)
    s_cts = add_multi(be, cts, y_cts)
    S_back = unpack_colmajor_multi(be, s_cts, L=L, hidden=hidden)
    err = np.max(np.abs(S_back - (X + Y)))
    log(f"  add err: {err:.3e}")
    assert err < 1e-5

    # 3. LayerNorm
    log("=== test layernorm_colmajor_multi ===")
    invsqrt_coeffs = [1.0, -0.5, 0.375]  # Taylor of 1/sqrt(1+u) at u=0
    invsqrt_interval = (-1.0, 1.0)
    gamma = np.ones(hidden)
    beta = np.zeros(hidden)
    t = time.time()
    ln_cts = layernorm_colmajor_multi(
        be, cts, L=L, hidden=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=invsqrt_interval,
        gamma=gamma, beta=beta,
    )
    dt = time.time() - t
    log(f"  LN multi-ct took {dt*1000:.0f} ms (depth={be._ops.depth(ln_cts[0])})")
    LN_back = unpack_colmajor_multi(be, ln_cts, L=L, hidden=hidden)

    # Reference: LN with same poly-invsqrt approximation
    mu = X.mean(axis=1, keepdims=True)
    centred = X - mu
    var = (centred ** 2).mean(axis=1, keepdims=True)
    # Map var into [-1,1] via 2/(b-a)*v + (-(a+b)/(b-a)) = 1*v + 0 here.
    # poly approximation: 1 - 0.5*u + 0.375*u^2
    u = var
    inv_sigma_approx = invsqrt_coeffs[0] + invsqrt_coeffs[1]*u + invsqrt_coeffs[2]*u**2
    LN_ref = centred * inv_sigma_approx  # gamma=1, beta=0
    err = np.max(np.abs(LN_back - LN_ref))
    log(f"  LN max err vs poly-ref: {err:.3e}")
    assert err < 1e-2, f"LN err too high: {err}"

    # 4. Linear (matches Wq shape: 768 → 768 in BERT-base)
    log("=== test linear_colmajor_multi_streaming (768→768) ===")
    Wq = rng.standard_normal((hidden, hidden)) * 0.05
    t = time.time()
    q_cts = linear_colmajor_multi_streaming(
        be, cts, Wq, L=L, in_dim=hidden, out_dim=hidden,
    )
    dt = time.time() - t
    log(f"  linear took {dt*1000:.0f} ms ({len(q_cts)} out cts, "
        f"depth={be._ops.depth(q_cts[0])})")
    Q_back = unpack_colmajor_multi(be, q_cts, L=L, hidden=hidden)
    Q_ref = X @ Wq.T
    err = np.max(np.abs(Q_back - Q_ref))
    log(f"  linear max err: {err:.3e}")
    assert err < 1e-4, f"linear err too high: {err}"

    # 5. Linear with different in/out shapes (W1 of FFN: 768 → 3072)
    log("=== test linear 768→3072 (FFN W1) ===")
    inter = 4 * hidden
    W1 = rng.standard_normal((inter, hidden)) * 0.05
    t = time.time()
    h_cts = linear_colmajor_multi_streaming(
        be, cts, W1, L=L, in_dim=hidden, out_dim=inter,
    )
    dt = time.time() - t
    log(f"  W1 took {dt*1000:.0f} ms ({len(h_cts)} out cts, expect 12 = 3072/256)")
    H_back = unpack_colmajor_multi(be, h_cts, L=L, hidden=inter)
    H_ref = X @ W1.T
    err = np.max(np.abs(H_back - H_ref))
    log(f"  W1 max err: {err:.3e}")
    assert err < 1e-3, f"W1 err too high: {err}"

    print()
    print("=" * 70)
    print(f"PASS — multi-ct primitives correct at L={L} N={N}")
    print("=" * 70)


if __name__ == "__main__":
    main()
