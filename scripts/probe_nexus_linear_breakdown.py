"""Decompose nexus_linear cost: encoding vs CtoS vs matvec vs gather.

Distinguishes (a) per-call overhead from (b) actual compute. Determines
whether the bottleneck is fixable by simple batching or needs the full
NEXUS compress/decompress algorithm.
"""
import time
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend


def main():
    print("Init HEonGPU N=2^16, 30-chain...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    be.configure_bootstrapping()
    print(f"  num_slots={be._num_slots}\n")

    # Realistic BERT-tiny attn shape (well within fast regime)
    in_dim, out_dim = 128, 128
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05

    # Warm-up: registers all needed galois keys
    print("Warm-up...")
    ct_x = be.encrypt_coeff(x.tolist())
    _ = be.nexus_linear(ct_x, W, in_dim=in_dim)
    print(f"  registered keys: {len(be._registered_shifts)}\n")

    # Decompose: time each stage separately
    print(f"=== Per-stage breakdown ({in_dim}x{out_dim}, median of 5) ===")
    n_rep = 5

    # 1. encrypt_coeff (encode + encrypt)
    times = []
    for _ in range(n_rep):
        t = time.time()
        _ = be.encrypt_coeff(x.tolist())
        times.append(time.time() - t)
    t_enc = np.median(times) * 1000
    print(f"  (1) encrypt_coeff           : {t_enc:.1f} ms")

    # 2. coeff_to_slot via nexus pipeline (CtoS + matvec)
    ct_x = be.encrypt_coeff(x.tolist())
    times = []
    for _ in range(n_rep):
        t = time.time()
        cts = be.coeff_matvec_to_slot(ct_x, W, in_dim=in_dim)
        times.append(time.time() - t)
    t_cts = np.median(times) * 1000
    print(f"  (2) coeff_matvec_to_slot    : {t_cts:.1f} ms")

    # 3. gather_slots
    targets = be.nexus_target_slots(in_dim, out_dim)
    cts = be.coeff_matvec_to_slot(ct_x, W, in_dim=in_dim)
    times = []
    for _ in range(n_rep):
        cc = cts[0]  # NB: gather_slots is destructive on its own clone
        t = time.time()
        _ = be.gather_slots(cc, targets)
        times.append(time.time() - t)
    t_gather = np.median(times) * 1000
    print(f"  (3) gather_slots            : {t_gather:.1f} ms")

    # 4. full nexus_linear
    times = []
    for _ in range(n_rep):
        ct_x = be.encrypt_coeff(x.tolist())
        t = time.time()
        _ = be.nexus_linear(ct_x, W, in_dim=in_dim)
        times.append(time.time() - t)
    t_full = np.median(times) * 1000
    print(f"  (4) full nexus_linear       : {t_full:.1f} ms")

    # 5. Lone primitive timings on N=2^16 ct
    print(f"\n=== Primitive ops on single ct (median of 10) ===")
    pt = be._encoder.encode(be._ctx, x.tolist() + [0.0]*(be._num_slots - in_dim), be._scale)
    ct = be._encryptor.encrypt(be._ctx, pt)

    times = []
    for _ in range(10):
        t = time.time()
        be._ops.multiply_plain_inplace(ct, pt)
        be._ops.rescale_inplace(ct)
        times.append(time.time() - t)
    print(f"  multiply_plain + rescale    : {np.median(times)*1000:.2f} ms")

    times = []
    for _ in range(10):
        t = time.time()
        be._ops.rotate_rows_inplace(ct, be._gk, 1)
        times.append(time.time() - t)
    print(f"  rotate_rows                 : {np.median(times)*1000:.2f} ms")

    times = []
    for _ in range(10):
        t = time.time()
        _ = be._encoder.encode(be._ctx, x.tolist() + [0.0]*(be._num_slots - in_dim), be._scale)
        times.append(time.time() - t)
    print(f"  encode plaintext            : {np.median(times)*1000:.2f} ms")

    times = []
    for _ in range(10):
        t = time.time()
        _ = be._encryptor.encrypt(be._ctx, pt)
        times.append(time.time() - t)
    print(f"  encrypt plaintext           : {np.median(times)*1000:.2f} ms")

    print("\n=== Conclusion ===")
    overhead_per_call = t_full - t_enc - t_cts - t_gather
    print(f"  Sum of (1)+(2)+(3)         : {t_enc + t_cts + t_gather:.1f} ms")
    print(f"  Full call                  : {t_full:.1f} ms")
    print(f"  Unaccounted overhead       : {overhead_per_call:.1f} ms ({overhead_per_call/t_full*100:.0f}%)")


if __name__ == "__main__":
    main()
