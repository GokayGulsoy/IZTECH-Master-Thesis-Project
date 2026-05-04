"""Phase 7e-1 smoke: validate NEXUS-style diagonal Q@K^T algorithm.

Hand-packs Q, K in column-major and runs `qk_scores_nexus`; compares
against a torch reference. Validates the algorithm in isolation before
we tackle projection-into-column-major (Phase 7e-3).
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor, qk_scores_nexus,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L, head_dim = 128, 64
    log(f"Config L={L} head_dim={head_dim}")

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

    rng = np.random.default_rng(11)
    Q = rng.standard_normal((L, head_dim)) * 0.1
    K = rng.standard_normal((L, head_dim)) * 0.1
    scale = 1.0 / np.sqrt(head_dim)

    # Reference: S[i, j] = scale * Σ_k Q[i, k] * K[j, k] = scale * Q @ K^T
    S_ref = scale * (Q @ K.T)
    log(f"  ref S shape {S_ref.shape}, range [{S_ref.min():.3f}, {S_ref.max():.3f}]")

    log("Pack Q, K column-major...")
    Q_ct = pack_colmajor(be, Q, L=L, head_dim=head_dim)
    K_ct = pack_colmajor(be, K, L=L, head_dim=head_dim)

    # Sanity: round-trip Q.
    Q_back = unpack_colmajor(be, Q_ct, L=L, head_dim=head_dim)
    rt_err = np.max(np.abs(Q_back - Q))
    log(f"  pack/unpack round-trip err: {rt_err:.2e}")

    log("Compute NEXUS Q@K^T...")
    t = time.time()
    S_ct = qk_scores_nexus(be, Q_ct, K_ct, L=L, head_dim=head_dim, scale=scale)
    wall = time.time() - t
    log(f"  wall = {wall*1000:.0f}ms  depth={be._ops.depth(S_ct)}")

    # S is in "diagonal-row" layout: slot[d*L + i] = S[i, (i+d) mod L]
    # Decrypt and rebuild full S matrix for comparison.
    slots = be.decrypt(S_ct)
    S_got = np.zeros((L, L), dtype=np.float64)
    for d in range(L):
        for i in range(L):
            S_got[i, (i + d) % L] = slots[d * L + i]

    err = np.max(np.abs(S_got - S_ref))
    log(f"  max err vs reference: {err:.3e}")
    if err > 1e-4:
        log("FAIL — error too large")
        log(f"  first row got:  {S_got[0, :8]}")
        log(f"  first row ref:  {S_ref[0, :8]}")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
