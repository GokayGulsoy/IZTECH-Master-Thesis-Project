"""Phase 7a-2: measure nexus_linear at NEXUS-recipe chain depths.

Hypothesis: NEXUS achieves 37s because matmul uses chain={60,40,60}=3 primes.
Our chain is 30 primes → every CUDA op is ~10x slower (more RNS limbs).

Test: mod-drop our ct to depth = max_depth - target, then time nexus_linear.
This approximates running on a shorter chain.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Init backend (chain=30)...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready. max_depth={be._max_depth}")
    log("configure_bootstrapping (needed for nexus_linear's CtoS) ...")
    be.configure_bootstrapping()
    log("  done")

    in_dim, out_dim = 128, 128
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05

    # Warm
    ct_x = be.encrypt_coeff(x.tolist())
    _ = be.nexus_linear(ct_x, W, in_dim=in_dim)

    print("\n=== nexus_linear time vs starting depth (level budget remaining) ===")
    print(f"{'starting_depth':>15s}  {'levels_left':>11s}  {'wall_ms':>10s}")
    for start_depth in [0, 5, 10, 15, 20, 25, 27]:
        ct_x = be.encrypt_coeff(x.tolist())
        # Mod-drop to start_depth
        for _ in range(start_depth):
            be._ops.mod_drop_inplace(ct_x)
        d = be._ops.depth(ct_x)
        # Time
        t = time.time()
        _ = be.nexus_linear(ct_x, W, in_dim=in_dim)
        dt = (time.time() - t) * 1000
        levels_left = be._max_depth - d
        print(f"{start_depth:>15d}  {levels_left:>11d}  {dt:>10.1f}")

    print("\n=== Interpretation ===")
    print("If nexus_linear gets MUCH faster at high start_depth (few levels left),")
    print("then the per-op cost is dominated by chain length and we should use")
    print("a shallower context for matmul (NEXUS recipe).")


if __name__ == "__main__":
    main()
