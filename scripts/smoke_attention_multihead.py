"""Phase 7e-6b smoke: multi-head packed NEXUS attention parity.

Validates that packing num_heads_per_ct heads side-by-side in one ct
produces the same result (per head) as running the kernel num_heads_per_ct
times serially with single-head input.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor,
    qk_scores_nexus, attn_apply_nexus,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 32
    head_dim = 64
    num_heads_per_ct = 4
    H = head_dim * L  # 2048

    log(f"Config L={L} head_dim={head_dim} num_heads_per_ct={num_heads_per_ct}")
    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 14,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready")

    rng = np.random.default_rng(42)
    Qs = [rng.standard_normal((L, head_dim)) * 0.1 for _ in range(num_heads_per_ct)]
    Ks = [rng.standard_normal((L, head_dim)) * 0.1 for _ in range(num_heads_per_ct)]
    Vs = [rng.standard_normal((L, head_dim)) * 0.1 for _ in range(num_heads_per_ct)]

    # Pack all heads into one wide col-major matrix of shape (L, num_heads*head_dim).
    Q_wide = np.concatenate(Qs, axis=1)
    K_wide = np.concatenate(Ks, axis=1)
    V_wide = np.concatenate(Vs, axis=1)
    log(f"  Q_wide shape {Q_wide.shape}")

    Q_ct = pack_colmajor(be, Q_wide, L=L, head_dim=num_heads_per_ct * head_dim)
    K_ct = pack_colmajor(be, K_wide, L=L, head_dim=num_heads_per_ct * head_dim)
    V_ct = pack_colmajor(be, V_wide, L=L, head_dim=num_heads_per_ct * head_dim)

    inv_sqrt_d = 1.0 / np.sqrt(head_dim)

    # Reference per-head softmax-poly + attention (using same poly approx).
    softmax_coeffs = [1.0, 0.5, 0.125]

    def py_softmax_poly(z):
        return sum(c * (z ** k) for k, c in enumerate(softmax_coeffs))

    log("Multi-head packed kernel...")
    t = time.time()
    S = qk_scores_nexus(be, Q_ct, K_ct, L=L, head_dim=head_dim,
                       scale=inv_sqrt_d, num_heads_per_ct=num_heads_per_ct)
    log(f"  qk wall = {(time.time()-t)*1000:.0f}ms")
    t = time.time()
    A = be.polyval(S, list(softmax_coeffs))
    log(f"  softmax wall = {(time.time()-t)*1000:.0f}ms")
    t = time.time()
    Out = attn_apply_nexus(be, A, V_ct, L=L, head_dim=head_dim,
                          num_heads_per_ct=num_heads_per_ct)
    log(f"  av wall = {(time.time()-t)*1000:.0f}ms")

    Out_wide = unpack_colmajor(be, Out, L=L, head_dim=num_heads_per_ct * head_dim)

    # Reference: do per-head separately.
    max_err = 0.0
    for h in range(num_heads_per_ct):
        Q, K, V = Qs[h], Ks[h], Vs[h]
        S_ref = (Q @ K.T) * inv_sqrt_d
        A_ref = py_softmax_poly(S_ref)
        Out_ref_h = A_ref @ V
        Out_got_h = Out_wide[:, h * head_dim:(h + 1) * head_dim]
        err = np.max(np.abs(Out_got_h - Out_ref_h))
        log(f"  head {h}: err {err:.3e}")
        max_err = max(max_err, err)

    log(f"  max err across heads: {max_err:.3e}")
    if max_err > 1e-4:
        log("FAIL")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
