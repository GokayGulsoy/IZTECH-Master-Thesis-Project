"""Primitive-op micro-benchmark on H100 via the HEonGPU pybind11 wrapper.

Counterpart to ``scripts/bench_ring_primitives.py`` (CPU OpenFHE). Run on
the Pod after building the wrapper:

    cd /workspace/repo
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \\
        python -m scripts.bench_heongpu_primitives --ring 32768 --depth 18

Prints per-op latency for encrypt, decrypt, mul_plain, mul (with relin),
add, rotate. Use to project the wall-clock cost of an ``enc_linear_matrix``
on H100 vs CPU.
"""

from __future__ import annotations

import argparse
import statistics
import time

from fhe_thesis.encryption.heongpu_bindings import (
    CKKSContext,
    Decryptor,
    Encoder,
    Encryptor,
    KeyGenerator,
    Operator,
)


def bench(label: str, fn, iters: int) -> float:
    # Warmup.
    fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(samples)
    p95 = sorted(samples)[int(0.95 * (iters - 1))]
    print(f"  {label:<14s} median={med:7.3f} ms   p95={p95:7.3f} ms   "
          f"(n={iters})")
    return med


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ring", type=int, default=32768,
                   help="poly_modulus_degree (N).")
    p.add_argument("--depth", type=int, default=18,
                   help="Number of intermediate prime levels for Q.")
    p.add_argument("--prime-bits", type=int, default=40)
    p.add_argument("--special-bits", type=int, default=60)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    Q_bits = [args.special_bits] + [args.prime_bits] * args.depth
    P_bits = [args.special_bits]

    print(f"=== HEonGPU primitives — N={args.ring}, depth={args.depth} ===")
    t0 = time.perf_counter()
    ctx = CKKSContext(args.ring, Q_bits, P_bits, sec_none=False)
    kg = KeyGenerator(ctx)
    sk = kg.generate_secret_key(ctx)
    pk = kg.generate_public_key(ctx, sk)
    rk = kg.generate_relin_key(ctx, sk)
    # Power-of-two shifts cover any rotation via decomposition.
    pow2_shifts = [(1 << i) for i in range(args.ring.bit_length() - 1)]
    pow2_shifts += [-s for s in pow2_shifts]
    gk = kg.generate_galois_key(ctx, sk, pow2_shifts)
    enc = Encoder(ctx)
    encryptor = Encryptor(ctx, pk)
    decryptor = Decryptor(ctx, sk)
    ops = Operator(ctx, enc)
    print(f"keygen+rotation-keys ({len(pow2_shifts)} shifts): "
          f"{time.perf_counter() - t0:.2f} s")

    slots = ctx.slot_count
    scale = 2.0**args.prime_bits
    msg = [0.5] * slots
    pt_vec = [0.25] * slots

    p1 = enc.encode(ctx, msg, scale)
    p_other = enc.encode(ctx, pt_vec, scale)

    bench("encrypt",   lambda: encryptor.encrypt(ctx, p1),                args.iters)
    c0 = encryptor.encrypt(ctx, p1)
    bench("decrypt",   lambda: decryptor.decrypt(ctx, c0),                args.iters)
    bench("mul_plain", lambda: _mul_plain(ops, encryptor, ctx, p1, p_other),
          args.iters)
    bench("mul+relin", lambda: _mul_relin(ops, encryptor, ctx, p1, rk),  args.iters)
    bench("rotate(1)", lambda: _rotate(ops, encryptor, ctx, p1, gk, 1),  args.iters)
    bench("rotate(64)", lambda: _rotate(ops, encryptor, ctx, p1, gk, 64), args.iters)
    bench("add",       lambda: _add(ops, encryptor, ctx, p1),            args.iters)

    # Projected matmul cost (pure rotation+mul_plain proxy).
    n = max(args.ring // 2, 1024)  # placeholder; user supplies via CLI in real bench
    print(f"\n(Heuristic projection of one BERT-Base linear, n=1024 diagonals:")
    print(f" ~= 1024 * (rotate + mul_plain + add). Use the medians above × 1024.)")
    return 0


def _mul_plain(ops, encryptor, ctx, p1, p_other):
    c = encryptor.encrypt(ctx, p1)
    ops.multiply_plain_inplace(c, p_other)
    return c


def _mul_relin(ops, encryptor, ctx, p1, rk):
    c1 = encryptor.encrypt(ctx, p1)
    c2 = encryptor.encrypt(ctx, p1)
    ops.multiply_inplace(c1, c2)
    ops.relinearize_inplace(c1, rk)
    return c1


def _rotate(ops, encryptor, ctx, p1, gk, shift):
    c = encryptor.encrypt(ctx, p1)
    ops.rotate_rows_inplace(c, gk, shift)
    return c


def _add(ops, encryptor, ctx, p1):
    c1 = encryptor.encrypt(ctx, p1)
    c2 = encryptor.encrypt(ctx, p1)
    ops.add_inplace(c1, c2)
    return c1


if __name__ == "__main__":
    import sys
    sys.exit(main())
