"""Decompose gather_slots cost: mask encoding vs mul_plain vs rotate."""
import time
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    N = be._num_slots
    print(f"N={N}\n")

    # Single ct + plaintext for primitive timings
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N) * 0.3
    pt = be._encoder.encode(be._ctx, x.tolist(), be._scale)
    ct = be._encryptor.encrypt(be._ctx, pt)

    # 1. Build mask list (Python overhead)
    times = []
    for _ in range(20):
        t = time.time()
        mask = [0.0] * N
        for p in [3, 7, 17, 42, 100]:
            mask[p] = 1.0
        times.append(time.time() - t)
    print(f"  build mask list (5 ones)   : {np.median(times)*1000:.3f} ms")

    # 2. Encode plaintext from mask list
    times = []
    for _ in range(20):
        mask = [0.0] * N
        for p in [3, 7, 17, 42, 100]:
            mask[p] = 1.0
        t = time.time()
        _ = be._encoder.encode(be._ctx, mask, be._scale)
        times.append(time.time() - t)
    print(f"  encode plaintext           : {np.median(times)*1000:.3f} ms")

    # 3. mul_plain alone (in-place, with cached pt)
    cached_pt = be._encoder.encode(be._ctx, [1.0]*5 + [0.0]*(N-5), be._scale)
    # warm
    be._ops.multiply_plain_inplace(ct, cached_pt)
    be._ops.rescale_inplace(ct)
    # fresh ct each iter (mul_plain is destructive)
    times = []
    for _ in range(20):
        ct2 = be.encrypt([0.5]*N)
        t = time.time()
        be._ops.multiply_plain_inplace(ct2, cached_pt)
        times.append(time.time() - t)
    print(f"  multiply_plain_inplace     : {np.median(times)*1000:.3f} ms")

    # 4. mul_plain via backend wrapper (does encode+mul+rescale+clone)
    times = []
    for _ in range(20):
        ct2 = be.encrypt([0.5]*N)
        t = time.time()
        _ = be.mul_plain(ct2, [1.0]*5 + [0.0]*(N-5))
        times.append(time.time() - t)
    print(f"  be.mul_plain (full path)   : {np.median(times)*1000:.3f} ms")

    print("\nIf mul_plain is fast but be.mul_plain is slow,")
    print("the overhead is in the wrapper (encode+clone).")


if __name__ == "__main__":
    main()
