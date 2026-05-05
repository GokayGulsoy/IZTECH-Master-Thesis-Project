"""Phase 8i smoke: streaming column-major linear correctness vs eager.

At L=32 the eager and streaming plans MUST produce identical results
(same compute pattern, just different plaintext lifetime). Verifies
the new C++ entry points (prepare_baby_rotations, accumulate_giant)
work correctly.
"""
import numpy as np
import time

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor,
    build_colmajor_linear_plan, linear_colmajor_bsgs_cpp,
    build_colmajor_linear_plan_streaming, linear_colmajor_streaming,
    prepare_colmajor_keys,
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L, in_dim, out_dim = 32, 768, 768
    log(f"L={L} in_dim={in_dim} out_dim={out_dim}")

    log("Init backend N=2^16...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 14,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth}")

    prepare_colmajor_keys(be, L=L, max_dim=1024)

    rng = np.random.default_rng(7)
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    x = rng.standard_normal((L, in_dim)) * 0.1
    Y_ref = x @ W.T

    log("Encrypt x...")
    x_ct = pack_colmajor(be, x, L=L, head_dim=in_dim)
    d0 = be._ops.depth(x_ct)
    log(f"  x depth={d0}")

    log("--- EAGER plan (existing) ---")
    t = time.time()
    plan_e = build_colmajor_linear_plan(be, W, L=L, in_dim=in_dim,
                                          out_dim=out_dim, ct_depth=d0)
    log(f"  build {(time.time()-t)*1000:.0f} ms")
    t = time.time()
    Y_e = linear_colmajor_bsgs_cpp(be, x_ct, plan_e)
    log(f"  exec {(time.time()-t)*1000:.0f} ms")
    Y_e_dec = unpack_colmajor(be, Y_e, L=L, head_dim=out_dim)
    err_e = np.max(np.abs(Y_e_dec - Y_ref))
    log(f"  max err vs torch: {err_e:.3e}")

    log("--- STREAMING plan (new) ---")
    t = time.time()
    plan_s = build_colmajor_linear_plan_streaming(
        be, W, L=L, in_dim=in_dim, out_dim=out_dim, ct_depth=d0,
    )
    log(f"  build {(time.time()-t)*1000:.0f} ms (no encoding)")
    t = time.time()
    Y_s = linear_colmajor_streaming(be, x_ct, plan_s)
    log(f"  exec {(time.time()-t)*1000:.0f} ms")
    Y_s_dec = unpack_colmajor(be, Y_s, L=L, head_dim=out_dim)
    err_s = np.max(np.abs(Y_s_dec - Y_ref))
    log(f"  max err vs torch: {err_s:.3e}")

    parity = np.max(np.abs(Y_e_dec - Y_s_dec))
    log(f"  eager vs streaming parity: {parity:.3e}")

    if err_s < 1e-3 and parity < 1e-6:
        log("PASS")
    else:
        log("FAIL")


if __name__ == "__main__":
    main()
