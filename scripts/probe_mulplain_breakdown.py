"""Microbench: where does linear_compressed's 1.46ms/mul_plain go?

Test 4 setups for one full linear (M=768 outputs, T=64 tokens):
A. Current: encode_coeff + mul_plain + rescale (per row)
B. Pre-encode all W rows once, then 768 mul_plain + rescale
C. Pre-encode + mul_plain (no rescale)
D. Just clone (overhead measurement)
"""
from __future__ import annotations
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import enc_compress


def encode_w_inner_for_d(backend, w_d: np.ndarray, d: int):
    N = backend._N
    enc = np.zeros(N, dtype=np.float64)
    enc[0] = float(w_d[0])
    for i in range(1, d):
        enc[N - i] = -float(w_d[i])
    return backend._encoder.encode_coeff(backend._ctx, enc.tolist(), backend._scale)


def main():
    N = 1 << 16
    d = 768
    T = 64
    M = 768

    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 6,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, d)) * 0.1
    W = rng.standard_normal((M, d)) * 0.05

    x_pad = np.zeros(N)
    for t in range(T):
        x_pad[t * d:(t + 1) * d] = X[t]
    ct_x = enc_compress(be, x_pad.tolist())

    # ---------- A. Current: encode + mul + rescale ----------
    t = time.time()
    out_a = []
    for i in range(M):
        pt = encode_w_inner_for_d(be, W[i], d)
        out_a.append(be._mul_plain_pt(ct_x, pt))
    dt_a = time.time() - t
    print(f"A. encode + mul + rescale     : {dt_a*1000:7.1f}ms ({dt_a*1e6/M:.1f}μs/row)")
    del out_a

    # ---------- A1. Just encode_coeff loop ----------
    t = time.time()
    pts_pre = [encode_w_inner_for_d(be, W[i], d) for i in range(M)]
    dt_pre = time.time() - t
    print(f"A1. pre-encode {M} W rows     : {dt_pre*1000:7.1f}ms ({dt_pre*1e6/M:.1f}μs/row)")

    # ---------- B. Pre-encoded: just mul + rescale ----------
    t = time.time()
    out_b = [be._mul_plain_pt(ct_x, pts_pre[i]) for i in range(M)]
    dt_b = time.time() - t
    print(f"B. pre-encoded mul + rescale  : {dt_b*1000:7.1f}ms ({dt_b*1e6/M:.1f}μs/row)")
    del out_b

    # ---------- C. mul only, no rescale ----------
    ops = be._ops
    t = time.time()
    out_c = []
    for pt in pts_pre:
        out = be._clone(ct_x)
        ops.multiply_plain_inplace(out, pt)
        out_c.append(out)
    dt_c = time.time() - t
    print(f"C. clone + mul (no rescale)   : {dt_c*1000:7.1f}ms ({dt_c*1e6/M:.1f}μs/row)")

    # ---------- D. Just clone ----------
    t = time.time()
    clones = [be._clone(ct_x) for _ in range(M)]
    dt_d = time.time() - t
    print(f"D. just clone                 : {dt_d*1000:7.1f}ms ({dt_d*1e6/M:.1f}μs/row)")
    del clones

    # ---------- E. mul_plain only (no clone, in-place pattern) ----------
    # Reuse one clone, time only the mul
    t = time.time()
    for pt in pts_pre:
        scratch = be._clone(ct_x)
        ops.multiply_plain_inplace(scratch, pt)
    dt_e = time.time() - t
    print(f"E. clone+mul (no save list)   : {dt_e*1000:7.1f}ms ({dt_e*1e6/M:.1f}μs/row)")

    # ---------- F. Rescale only on results from C ----------
    t = time.time()
    for ct in out_c:
        ops.rescale_inplace(ct)
    dt_f = time.time() - t
    print(f"F. rescale {M} cts             : {dt_f*1000:7.1f}ms ({dt_f*1e6/M:.1f}μs/row)")

    print(f"\nBreakdown of B (mul+rescale): mul={dt_c*1000:.1f}ms, rescale={dt_f*1000:.1f}ms, sum={dt_c*1000+dt_f*1000:.1f}ms")
    print(f"Pre-encode cost is amortized if W is reused across multiple inputs (yes for inference)")


if __name__ == "__main__":
    main()
