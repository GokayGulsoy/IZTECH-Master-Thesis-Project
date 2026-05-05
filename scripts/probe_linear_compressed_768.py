"""Probe NEXUS linear_compressed at BERT-base widths.

linear_compressed is the M·mul_plain variant — input stays compressed,
no decompression, output is M cts each holding scalar at coeff[0].

Test:
- N = 2^16 (= 65536)
- Wq shape: 768x768. With NEXUS layout, we pad inputs to N=65536,
  weight rows are length-N vectors (most zero except first 768).
- Single linear cost = M ct·pt mults = 768 mul_plain.
"""
from __future__ import annotations

import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import (
    enc_compress, linear_compressed, fold_outputs_to_packed,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    N = 1 << 16
    log(f"Init N={N}")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 6,   # short chain, no bootstrap needed for one mul
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  num_slots={be._num_slots}  N={be._N}  max_depth={be._max_depth}")

    in_dim = 768
    out_dim = 768

    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim) * 0.3
    W = rng.standard_normal((out_dim, in_dim)) * 0.05
    expected = W @ x

    # Pad x to length N for compression.
    x_padded = np.zeros(N)
    x_padded[:in_dim] = x

    # Pad W to (out_dim, N): each row's first in_dim entries are W[i,:], rest zero.
    W_padded = np.zeros((out_dim, N))
    W_padded[:, :in_dim] = W

    log("Compress x (encrypt as polynomial coefficients)...")
    t = time.time()
    ct_x = enc_compress(be, x_padded.tolist())
    log(f"  done in {(time.time()-t)*1000:.1f}ms")

    log(f"Run linear_compressed (M={out_dim} ct·pt mults)...")
    times = []
    for rep in range(2):
        t = time.time()
        out_cts = linear_compressed(be, W_padded, ct_x)
        dt = time.time() - t
        times.append(dt)
        log(f"  rep{rep}: {dt*1000:.1f}ms ({dt/out_dim*1000:.2f}ms per output row)")

    # Validate first 8 outputs.
    log("\nValidation:")
    for i in [0, 1, 7, 100, 767]:
        pt = be._decryptor.decrypt(be._ctx, out_cts[i])
        decoded = np.array(be._encoder.decode(pt))
        got = decoded.mean()
        err = abs(got - expected[i])
        log(f"  i={i}  expect={expected[i]:+.4f}  got={got:+.4f}  err={err:.2e}")

    best = min(times)
    log(f"\n=== TIMING ===")
    log(f"  768x768 linear: {best*1000:.1f}ms (single input)")
    log(f"  Per layer (4·QKVO + W1[3072x768] + W2[768x3072]):")
    qkvo = 4 * best                                           # 4*768 = 3072 mul_plains
    w1 = best * (3072 / out_dim)                              # 3072 mul_plains
    w2 = best * (768 / out_dim) * (3072 / in_dim)             # 768 mul_plains, in=3072
    # Actually for W2: out=768, in=3072. M=768 mul_plains regardless of in_dim
    # because each weight row is just length-N. So same time as 768x768.
    w2 = best
    log(f"    QKVO ({4*768}=3072 muls):  {qkvo*1000:8.1f}ms")
    log(f"    W1 (3072 muls):           {w1*1000:8.1f}ms  (4 of these in time)")
    log(f"    W2 (768 muls):            {w2*1000:8.1f}ms")
    log(f"  Linear/layer ≈ {(qkvo + w1 + w2)*1000:8.1f}ms")
    log(f"  12-layer linear: {12*(qkvo+w1+w2):8.1f}s (single input)")
    log(f"  + non-linear/bootstrap overhead")


if __name__ == "__main__":
    main()
