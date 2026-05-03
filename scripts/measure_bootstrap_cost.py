"""Time bootstrap cost on a slot-encoded ciphertext."""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(be._num_slots) * 0.3
    ct = be.encrypt(x.tolist())

    # Drop to bottom
    while be._ops.depth(ct) < be._max_depth:
        be._ops.mod_drop_inplace_ct(ct)

    # warmup
    _ = be._ops.regular_bootstrapping(be._clone(ct), be._gk, be._rk)
    times = []
    for _ in range(3):
        ct_d = be._clone(ct)
        while be._ops.depth(ct_d) < be._max_depth:
            be._ops.mod_drop_inplace_ct(ct_d)
        t = time.time()
        _ = be._ops.regular_bootstrapping(ct_d, be._gk, be._rk)
        times.append(time.time() - t)
    print(f"  bootstrap (slot): median={np.median(times)*1000:.1f}ms  min={min(times)*1000:.1f}ms")
    print(f"  max_depth={be._max_depth}")


if __name__ == "__main__":
    main()
