"""Probe multi-token amortization via NEXUS coeff packing.

Setup:
- Pack T tokens of dim d=768 into ONE ct: x[t*d + i] = X[t, i]
  (T must satisfy T*d <= N. At d=768, N=65536 → T_max = 85)
- Encode w (length d) as inner-product weight: enc_w[0]=w[0], enc_w[k]=-w[d-k] for k=1..d-1
- ONE mul_plain produces coeff[t*d] = (X[t,:] @ w) for ALL t simultaneously

Verify: T=64 tokens × 768 hidden, 1 mul_plain == 64 dot products.
Then time M=768 mul_plains (one full linear) = 64 token outputs × 768 dims.
Compare per-token amortized cost.
"""
from __future__ import annotations
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import enc_compress


def encode_w_inner_for_d(backend, w_d: np.ndarray, d: int):
    """Encode w of length d so coeff[t*d] of (x_packed * enc_w) = X[t,:] @ w_d.

    For position 0 within a token-block: needs (i+j) ≡ 0 (mod N) for i in [0,d).
    With x packed at positions [t*d, t*d+d), j=N-i (i>0, neg sign) and j=0 (i=0).
    Same as standard inner-product encoding, just truncated:
      enc[0]=w_d[0], enc[N-i]=-w_d[i] for i=1..d-1, rest 0.
    """
    N = backend._N
    enc = np.zeros(N, dtype=np.float64)
    enc[0] = float(w_d[0])
    for i in range(1, d):
        enc[N - i] = -float(w_d[i])
    return backend._encoder.encode_coeff(backend._ctx, enc.tolist(), backend._scale)


def main():
    N = 1 << 16
    d = 768
    T = 64                      # tokens packed per ct (T*d = 49152 < N=65536)
    M = 768                     # output rows for the linear

    print(f"N={N}  d={d}  T={T}  M={M}")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 6,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, d)) * 0.1     # (T, d)
    W = rng.standard_normal((M, d)) * 0.05    # (M, d)
    expected = X @ W.T                        # (T, M)

    # Pack X into one length-N vector: positions t*d + i.
    x_pad = np.zeros(N)
    for t in range(T):
        x_pad[t * d:(t + 1) * d] = X[t]

    print("Compress X (single ct holds T tokens)...")
    ct_x = enc_compress(be, x_pad.tolist())
    print("  done")

    # ----- Single-output sanity check on row 0 -----
    print("\n--- Sanity: 1 mul_plain → T token outputs at coeff[t*d] ---")
    pt_w0 = encode_w_inner_for_d(be, W[0], d)
    out0 = be._mul_plain_pt(ct_x, pt_w0)
    decoded = np.array(be._encoder.decode(be._decryptor.decrypt(be._ctx, out0)))
    print(f"  expected[t=0..3, m=0]   = {expected[:4, 0].round(4)}")
    print(f"  decoded[t=0..3, *d]     = {decoded[[0, d, 2*d, 3*d]].round(4)}")
    err = max(abs(decoded[t * d] - expected[t, 0]) for t in range(T))
    print(f"  max err over T={T} tokens: {err:.3e}")
    if err > 1e-2:
        print("  FAIL — amortization layout broken")
        return
    print("  PASS")

    # ----- Time M=768 mul_plains (one full linear) -----
    print(f"\n--- Timing: M={M} mul_plains for full {M} outputs × {T} tokens ---")
    t = time.time()
    out_cts = []
    for i in range(M):
        pt_w = encode_w_inner_for_d(be, W[i], d)
        out_cts.append(be._mul_plain_pt(ct_x, pt_w))
    dt = time.time() - t
    print(f"  full linear: {dt*1000:.1f}ms")
    print(f"  per-token-output amortized: {dt*1000/(M*T):.4f}ms ({dt*1e6/(M*T):.1f}μs)")

    # Validate one random output ct
    rand_i = 100
    decoded = np.array(be._encoder.decode(be._decryptor.decrypt(be._ctx, out_cts[rand_i])))
    errs = [abs(decoded[t * d] - expected[t, rand_i]) for t in range(T)]
    print(f"  output i={rand_i}: max err over {T} tokens = {max(errs):.3e}")

    # ----- Project per-layer / per-12-layer cost -----
    seq = 128
    cts_per_seq = (seq * d + N - 1) // N      # = 2 (128*768 = 98304 → 2 cts)
    print(f"\n--- Projection (seq={seq}, d={d}) ---")
    print(f"  Input takes {cts_per_seq} cts (T_eff = {seq // cts_per_seq} per ct)")
    per_linear_per_ct = dt   # M mul_plains, T tokens
    # Per BERT layer: 4 (QKVO) + 2 (FFN W1, W2) linears, dimensions vary.
    # QKVO: M=768, in=768, T=T per ct, cts_per_seq cts → 4 * cts_per_seq * dt
    qkvo = 4 * cts_per_seq * dt
    # W1: M=3072, in=768, cts_per_seq cts. M is 4x → 4*cts_per_seq*dt = same as 1 of QKVO times 4? No: M scales linearly.
    w1 = (3072 / M) * cts_per_seq * dt
    # W2: M=768, in=3072. Cts_per_seq for in=3072: ceil(128*3072/N)=6
    cts_w2_in = (seq * 3072 + N - 1) // N
    w2 = (768 / M) * cts_w2_in * dt
    layer = qkvo + w1 + w2
    print(f"  QKVO    ≈ {qkvo:.2f}s   (4 linears × {cts_per_seq} input cts)")
    print(f"  FFN W1  ≈ {w1:.2f}s   (3072/{M}× × {cts_per_seq} input cts)")
    print(f"  FFN W2  ≈ {w2:.2f}s   (768/{M}× × {cts_w2_in} input cts)")
    print(f"  ----  per layer (linears only): {layer:.2f}s")
    print(f"  ----  12 layers (linears only): {12 * layer:.1f}s")
    print(f"  (excludes: attention scores, LN/softmax/GELU, bootstrap, transitions)")


if __name__ == "__main__":
    main()
