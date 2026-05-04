"""Phase 7e-3 smoke: column-major linear projection parity.

Validates linear_colmajor against W @ X^T reference.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor, linear_colmajor,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # BERT-base hidden=768 head_dim=64. L=32 fits in N=2^16 (768*32=24576 ≤ 32768).
    L, in_dim, out_dim = 32, 768, 64
    log(f"Config L={L} in_dim={in_dim} out_dim={out_dim}")

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

    rng = np.random.default_rng(17)
    X = rng.standard_normal((L, in_dim)) * 0.1
    W = rng.standard_normal((out_dim, in_dim)) * 0.1
    b = rng.standard_normal((out_dim,)) * 0.05

    Y_ref = X @ W.T + b
    log(f"  Y ref shape {Y_ref.shape} range [{Y_ref.min():.3f}, {Y_ref.max():.3f}]")

    log("Pack X col-major...")
    X_ct = pack_colmajor(be, X, L=L, head_dim=in_dim)

    log("Linear col-major...")
    t = time.time()
    Y_ct = linear_colmajor(be, X_ct, W, L=L, in_dim=in_dim, out_dim=out_dim, bias=b)
    log(f"  wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(Y_ct)}")

    Y_got = unpack_colmajor(be, Y_ct, L=L, head_dim=out_dim)
    err = np.max(np.abs(Y_got - Y_ref))
    log(f"  max err: {err:.3e}")
    if err > 1e-4:
        log("FAIL")
        log(f"  got[0]: {Y_got[0, :5]}")
        log(f"  ref[0]: {Y_ref[0, :5]}")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
