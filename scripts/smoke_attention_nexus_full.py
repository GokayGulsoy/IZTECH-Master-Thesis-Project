"""Phase 7e-2 smoke: full NEXUS attention (QK^T + softmax + AV) parity.

Validates the complete attention forward in column-major NEXUS layout
against a torch reference.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor, qk_scores_nexus, attn_apply_nexus,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L, head_dim = 32, 32
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

    rng = np.random.default_rng(13)
    Q = rng.standard_normal((L, head_dim)) * 0.1
    K = rng.standard_normal((L, head_dim)) * 0.1
    V = rng.standard_normal((L, head_dim)) * 0.1
    scale = 1.0 / np.sqrt(head_dim)

    # Reference using LPAN softmax poly (degree-2): softmax(S) ≈ 1 + 0.5 S + 0.125 S^2.
    sm_coeffs = [1.0, 0.5, 0.125]
    S_ref = scale * (Q @ K.T)
    A_ref = np.zeros_like(S_ref)
    for k, c in enumerate(sm_coeffs):
        A_ref += c * (S_ref ** k)
    Out_ref = A_ref @ V
    log(f"  Out ref shape {Out_ref.shape}, range [{Out_ref.min():.3f}, {Out_ref.max():.3f}]")

    log("Pack Q, K, V column-major...")
    Q_ct = pack_colmajor(be, Q, L=L, head_dim=head_dim)
    K_ct = pack_colmajor(be, K, L=L, head_dim=head_dim)
    V_ct = pack_colmajor(be, V, L=L, head_dim=head_dim)

    log("QK^T...")
    t = time.time()
    S_ct = qk_scores_nexus(be, Q_ct, K_ct, L=L, head_dim=head_dim, scale=scale)
    log(f"  wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(S_ct)}")

    log("Softmax poly (slot-local polyval)...")
    t = time.time()
    A_ct = be.polyval(S_ct, sm_coeffs)
    log(f"  wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(A_ct)}")

    log("A @ V...")
    t = time.time()
    Out_ct = attn_apply_nexus(be, A_ct, V_ct, L=L, head_dim=head_dim)
    log(f"  wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(Out_ct)}")

    Out_got = unpack_colmajor(be, Out_ct, L=L, head_dim=head_dim)
    err = np.max(np.abs(Out_got - Out_ref))
    log(f"  max err vs reference: {err:.3e}")
    if err > 1e-3:
        log("FAIL")
        log(f"  got[0]: {Out_got[0, :5]}")
        log(f"  ref[0]: {Out_ref[0, :5]}")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
