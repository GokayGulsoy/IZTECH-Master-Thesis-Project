"""Phase 7a-3: per-op cost vs chain depth.

Measures rotate, mul_plain, and add cost on cts at various depths.
This tells us whether shrinking the chain (NEXUS recipe) actually helps.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def median_ms(fn, n=10):
    ts = []
    for _ in range(n):
        t = time.time()
        fn()
        ts.append(time.time() - t)
    return float(np.median(ts)) * 1000


def main():
    log("Init chain=30 backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready. max_depth={be._max_depth}, num_slots={be._num_slots}")

    rng = np.random.default_rng(0)
    vec = rng.standard_normal(be._num_slots) * 0.3

    print(f"\n{'depth':>6s}  {'levels_left':>12s}  {'rotate_ms':>10s}  {'mul_plain_ms':>13s}  {'add_ms':>8s}  {'rescale_ms':>11s}")
    for target_depth in [0, 5, 10, 15, 20, 25, 28]:
        ct = be.encrypt(vec.tolist())
        for _ in range(target_depth):
            be._ops.mod_drop_inplace_ct(ct)
        d = be._ops.depth(ct)
        levels_left = be._max_depth - d

        # rotate
        def f_rot():
            _ = be._ops.rotate(ct, be._gk, 1)
        t_rot = median_ms(f_rot)

        # mul_plain (depth-0 plaintext, encoded fresh; need pt at same depth as ct)
        pt = be._encode_full(list(vec))
        # mod-drop the plaintext to ct's depth
        while be._ops.depth_of_plaintext(pt) < d:
            be._ops.mod_drop_inplace_pt(pt)

        def f_mul():
            ct2 = be._ops.clone_ct(ct)
            be._ops.multiply_plain_inplace(ct2, pt)
            # don't rescale — measure mul only
        t_mul = median_ms(f_mul)

        # add
        ct_b = be._ops.clone_ct(ct)
        def f_add():
            ct2 = be._ops.clone_ct(ct)
            be._ops.add_inplace(ct2, ct_b)
        t_add = median_ms(f_add)

        # rescale (need to be after a mul to make rescale legal)
        def f_rs():
            ct2 = be._ops.clone_ct(ct)
            be._ops.multiply_plain_inplace(ct2, pt)
            be._ops.rescale_inplace(ct2)
        t_rs = median_ms(f_rs) - t_mul  # subtract the mul cost

        print(f"{d:>6d}  {levels_left:>12d}  {t_rot:>10.2f}  {t_mul:>13.2f}  {t_add:>8.2f}  {t_rs:>11.2f}")


if __name__ == "__main__":
    main()
