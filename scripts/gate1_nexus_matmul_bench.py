"""GATE 1: Replicate NEXUS's own matmul benchmark at production N=2^16.

NEXUS test shape (from /workspace/NEXUS/cuda/src/main.cu):
  LHS = matrix_4096x768  (plaintext, 4096 rows in slots, 768 cols)
  RHS = matrix_768x64    (compressed cipher, 768 rows, 64 cols)
  → encoded as ⌈768*64/N⌉ compressed cts → decompressed to 768*64 broadcast cts
  Output = 64 cts × 4096 slot values each
  Cost: K*N = 768*64 = 49152 mul_plains + decompression of all 49K broadcast cts

Map to mm_nexus.py:
  matrix_mul(W, decompressed_x):
    W is (M, Ndim) plain — Ndim = #broadcast cts (must equal len(decompressed_x))
    M is output dim
  In NEXUS: M=64 outputs, Ndim=768 inner contracted dim per "decompressed group"

But mm_nexus.matrix_mul iterates per OUTPUT i, summing over ALL j.
NEXUS test does this 64 times (one per output column).

We measure:
  - Compress time (RHS encoding)
  - Decompress time (1 ct → N broadcast cts)
  - Per-output-row mul_plain accumulator time
  - Total wall
  - GPU memory peak (rough)
"""
from __future__ import annotations
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import (
    enc_compress, decompress, required_galois_elts,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    N = 1 << 16
    log(f"=== GATE 1: NEXUS matmul shape at N={N} (production) ===")

    # NEXUS shapes
    M_lhs = 4096       # LHS rows = batch*tokens (32*128 in NEXUS test)
    K = 768            # contracted dim = d_hidden
    N_out = 64         # output cols (= head_dim per matmul block)

    log(f"LHS: ({M_lhs}, {K}) plaintext  -- {M_lhs*K:,} elements")
    log(f"RHS: ({K}, {N_out}) cipher     -- {K*N_out:,} elements")
    log(f"OUT: ({M_lhs}, {N_out}) cipher -- {M_lhs*N_out:,} elements (in {N_out} cts of {M_lhs} slots)")

    log("Init HEonGPU backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 6,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  num_slots={be._num_slots}  N={be._N}  max_depth={be._max_depth}")

    rng = np.random.default_rng(0)
    LHS = rng.standard_normal((M_lhs, K)) * 0.1
    RHS = rng.standard_normal((K, N_out)) * 0.05
    expected = LHS @ RHS                      # (M_lhs, N_out)
    log(f"Reference computed: expected shape {expected.shape}")

    # ----- Encode LHS as K plaintexts of M_lhs slot values -----
    # Each plaintext encodes column j of LHS in slots [0, M_lhs).
    log(f"\n--- Encoding LHS as {K} slot-plaintexts ({M_lhs} slots each, padded to {be._num_slots}) ---")
    t = time.time()
    lhs_pts = []
    for j in range(K):
        slots = np.zeros(be._num_slots)
        slots[:M_lhs] = LHS[:, j]
        pt = be._encoder.encode(be._ctx, slots.tolist(), be._scale)
        lhs_pts.append(pt)
    dt = time.time() - t
    log(f"  encoded {K} pts in {dt:.2f}s ({dt*1000/K:.1f}ms each)")

    # ----- Compress RHS -----
    # Layout: pack RHS in COLUMN-MAJOR by output-col: [n*K + k] = RHS[k, n]
    log(f"\n--- Compressing RHS ({K*N_out} elements) ---")
    rhs_flat = np.zeros(N)
    for n in range(N_out):
        for k in range(K):
            rhs_flat[n * K + k] = RHS[k, n]
    t = time.time()
    ct_rhs = enc_compress(be, rhs_flat.tolist())
    log(f"  compressed in {(time.time()-t)*1000:.1f}ms (1 ct, {N} coeffs)")

    # ----- Generate Galois keys for decompression -----
    log("\n--- Generating Galois keys ---")
    t = time.time()
    elts = required_galois_elts(N)
    gk = be._kg.generate_galois_key_elts(be._ctx, be._sk, elts)
    log(f"  generated {len(elts)} galois elements in {time.time()-t:.2f}s")

    # ----- Decompress -----
    log(f"\n--- Decompressing 1 ct → {N} broadcast cts ({N//N_out:.0f} per output col) ---")
    log(f"    expected memory: {N} cts × ~{4 * (be._max_depth+1) * N * 8 / 1024 / 1024:.1f}MB  ≈  {N * 4 * (be._max_depth+1) * N * 8 / 1024**3:.0f} GB")
    t = time.time()
    try:
        decomp = decompress(be, ct_rhs, gk)
    except Exception as e:
        log(f"  DECOMPRESS FAILED: {type(e).__name__}: {e}")
        log("  GATE 1 BLOCKED on memory. Cannot proceed.")
        return
    dt_decomp = time.time() - t
    log(f"  decompressed {len(decomp)} cts in {dt_decomp:.2f}s")

    # ----- Matmul: per output col n, sum_k LHS_pt[k] * decomp[n*K + k] -----
    log(f"\n--- Matrix mul: {N_out} output cts × {K} mul_plain each = {N_out*K:,} ops ---")
    ops = be._ops
    out_cts = []
    t = time.time()
    for n in range(N_out):
        # Accumulator: start with k=0
        d0 = decomp[n * K + 0]
        acc = be._clone(d0)
        ops.multiply_plain_inplace(acc, lhs_pts[0])
        for k in range(1, K):
            tmp = be._clone(decomp[n * K + k])
            ops.multiply_plain_inplace(tmp, lhs_pts[k])
            ops.add_inplace_match(acc, tmp)
        ops.rescale_inplace(acc)
        out_cts.append(acc)
    dt_mm = time.time() - t
    log(f"  matmul: {dt_mm:.2f}s ({dt_mm*1000/(N_out*K):.3f}ms per mul_plain)")

    # ----- Validate first column -----
    log("\n--- Validate first column ---")
    pt = be._decryptor.decrypt(be._ctx, out_cts[0])
    decoded = np.array(be._encoder.decode(pt))
    # Decoded should be slot-domain (?). NEXUS divides by 4096 (= a "scale").
    # For our setting: decompressed cts at coeff scale, after mul scale * coeff_scale.
    # The factor of N from compression must be undone — check both interpretations.
    print(f"  decoded[:6]  = {decoded[:6]}")
    print(f"  expected[:6, 0] = {expected[:6, 0]}")
    print(f"  decoded[:6] / N = {decoded[:6] / N}")
    err_a = np.abs(decoded[:M_lhs] - expected[:, 0]).mean()
    err_b = np.abs(decoded[:M_lhs] / N - expected[:, 0]).mean()
    err_c = np.abs(decoded[:M_lhs] - expected[:, 0] / N).mean()
    log(f"  mean err (raw): {err_a:.3e}")
    log(f"  mean err (decoded/N): {err_b:.3e}")
    log(f"  mean err (expected/N): {err_c:.3e}")

    log(f"\n=== GATE 1 SUMMARY ===")
    log(f"  shape: ({M_lhs}, {K}) @ ({K}, {N_out}) = ({M_lhs}, {N_out})")
    log(f"  decompress: {dt_decomp:.2f}s")
    log(f"  matmul:     {dt_mm:.2f}s")
    log(f"  TOTAL:      {dt_decomp + dt_mm:.2f}s")
    log(f"  this matmul amortizes {M_lhs * N_out:,} cleartext mul-adds")
    log(f"  per-output-element cost: {(dt_decomp+dt_mm) * 1e6 / (M_lhs * N_out):.2f}μs")


if __name__ == "__main__":
    main()
