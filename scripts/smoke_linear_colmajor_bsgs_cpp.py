"""Phase 7e-7 smoke: C++ BSGS column-major linear parity + bench.

Validates linear_colmajor_bsgs_cpp against the Python linear_colmajor.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor,
    linear_colmajor, linear_colmajor_bsgs_cpp,
    build_colmajor_linear_plan, prepare_colmajor_keys,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
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
    log(f"  Y ref range [{Y_ref.min():.3f}, {Y_ref.max():.3f}]")

    log("Pre-register rotation keys...")
    in_dim_padded = 1
    while in_dim_padded < in_dim:
        in_dim_padded <<= 1
    n_new = prepare_colmajor_keys(be, L=L, max_dim=in_dim_padded)
    log(f"  +{n_new} keys")

    log("Pack X col-major...")
    X_ct = pack_colmajor(be, X, L=L, head_dim=in_dim)
    log(f"  X_ct depth={be._ops.depth(X_ct)}")

    # Reference: existing Python BSGS
    log("Python BSGS linear_colmajor (warm)...")
    _ = linear_colmajor(be, X_ct, W, L=L, in_dim=in_dim, out_dim=out_dim, bias=b)  # warm
    t = time.time()
    Y_py = linear_colmajor(be, X_ct, W, L=L, in_dim=in_dim, out_dim=out_dim, bias=b)
    log(f"  Python wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(Y_py)}")

    log("Build C++ BSGS plan (one-off)...")
    t = time.time()
    plan = build_colmajor_linear_plan(
        be, W, L=L, in_dim=in_dim, out_dim=out_dim, bias=b,
        ct_depth=be._ops.depth(X_ct),
    )
    log(f"  build wall = {(time.time()-t)*1000:.0f}ms  (encoded {len(plan.bucket_masks)} diagonals)")

    log("C++ BSGS execute (cold)...")
    t = time.time()
    Y_cpp = linear_colmajor_bsgs_cpp(be, X_ct, plan)
    log(f"  C++ wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(Y_cpp)}")

    log("C++ BSGS execute (warm)...")
    t = time.time()
    Y_cpp = linear_colmajor_bsgs_cpp(be, X_ct, plan)
    log(f"  C++ wall = {(time.time()-t)*1000:.0f}ms  depth={be._ops.depth(Y_cpp)}")

    Y_py_dec  = unpack_colmajor(be, Y_py,  L=L, head_dim=out_dim)
    Y_cpp_dec = unpack_colmajor(be, Y_cpp, L=L, head_dim=out_dim)

    err_py  = np.max(np.abs(Y_py_dec  - Y_ref))
    err_cpp = np.max(np.abs(Y_cpp_dec - Y_ref))
    err_match = np.max(np.abs(Y_py_dec - Y_cpp_dec))

    log(f"  Python err vs ref:  {err_py:.3e}")
    log(f"  C++    err vs ref:  {err_cpp:.3e}")
    log(f"  Python vs C++ diff: {err_match:.3e}")

    if max(err_py, err_cpp) > 1e-4:
        log("FAIL")
        raise SystemExit(1)
    log("PASS")


if __name__ == "__main__":
    main()
