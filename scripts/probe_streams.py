"""NEXUS Phase 5 micro-benchmark: parallel BSGS-matmul on N streams.

Workload: 4 independent matmul_plain operations (mimicking QKVO).
Each consists of (n) cycles of: rotate + mul_plain + add.
Compare:
  (A) Serial: 4 matmuls back-to-back on default stream
  (B) Parallel: 4 matmuls on 4 streams via Python threads

Speedup tells us if it's worth wiring streams into the protocol.
"""
import time
import numpy as np
import threading
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption import heongpu_bindings as he


def matmul_workload(be, ct, w_diags, gk, stream=None):
    """Mimic BSGS matmul: replicate-in-block + n_diag rotations + mul + add.

    For benchmarking only; not numerically meaningful.
    """
    op = be._ops
    if stream is None:
        # serial path uses default stream (existing inplace methods)
        result = op.clone_ct(ct)
        op.multiply_plain_inplace(result, w_diags[0])
        for diag in w_diags[1:]:
            x = op.clone_ct(ct)
            op.rotate_rows_inplace(x, gk, 1)
            op.multiply_plain_inplace(x, diag)
            op.add_inplace(result, x)
        return result
    else:
        result = op.clone_ct_s(ct, stream)
        op.multiply_plain_inplace_s(result, w_diags[0], stream)
        for diag in w_diags[1:]:
            x = op.clone_ct_s(ct, stream)
            op.rotate_rows_inplace_s(x, gk, 1, stream)
            op.multiply_plain_inplace_s(x, diag, stream)
            op.add_inplace_s(result, x, stream)
        return result


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
    rng = np.random.default_rng(7)
    num_slots = be._num_slots

    # Mimic typical attention QKVO: hidden dim 768 → BSGS uses ~28 baby
    # diagonals in production. We test with n_diag=64 to be conservative.
    n_diag = 64
    print(f"  num_slots={num_slots}  n_diag per matmul={n_diag}")

    # Build common: input ct + 4 weight matrices (each as a list of n_diag plaintexts)
    x = rng.standard_normal(num_slots) * 0.3
    ct = be.encrypt(x.tolist())

    weights = []
    for k in range(4):
        diag_pts = []
        for i in range(n_diag):
            d = rng.standard_normal(num_slots) * 0.05
            pt = be._encoder.encode(be._ctx, d.tolist(), be._scale)
            diag_pts.append(pt)
        weights.append(diag_pts)
    print(f"  encoded 4 × {n_diag} weight diagonals")

    # Warm-up: registers galois key for shift=1 (already pre-gen'd in backend)
    _ = matmul_workload(be, ct, weights[0], be._gk, stream=None)

    # ── (A) Serial baseline ──
    n_repeats = 3
    times = []
    for _ in range(n_repeats):
        t0 = time.time()
        results = [matmul_workload(be, ct, weights[k], be._gk, stream=None)
                   for k in range(4)]
        times.append(time.time() - t0)
    serial = float(np.median(times))
    print(f"\n  (A) SERIAL 4 matmuls (default stream): {serial*1000:.1f}ms (median of {n_repeats})")

    # ── (B) Parallel via Python threads + 4 streams ──
    streams = [he.CudaStream() for _ in range(4)]

    def worker(k, out_list):
        out_list[k] = matmul_workload(be, ct, weights[k], be._gk, stream=streams[k])

    # Warm-up
    out_warm = [None] * 4
    threads = [threading.Thread(target=worker, args=(k, out_warm)) for k in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    for s in streams: s.synchronize()

    times = []
    for _ in range(n_repeats):
        out = [None] * 4
        threads = [threading.Thread(target=worker, args=(k, out)) for k in range(4)]
        t0 = time.time()
        for t in threads: t.start()
        for t in threads: t.join()
        for s in streams: s.synchronize()
        times.append(time.time() - t0)
    parallel = float(np.median(times))
    print(f"  (B) PARALLEL 4 matmuls (4 streams + threads): {parallel*1000:.1f}ms")

    speedup = serial / parallel
    print(f"\n  Speedup: {speedup:.2f}×")
    if speedup >= 1.5:
        print(f"  ✅ Worth wiring into protocol — would save ~{(1 - 1/speedup)*100:.0f}% of QKVO wall-time")
    else:
        print(f"  ⚠️  Marginal speedup; GPU compute likely already saturating SMs")


if __name__ == "__main__":
    main()
