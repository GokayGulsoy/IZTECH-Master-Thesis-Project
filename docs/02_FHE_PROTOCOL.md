# 02 — FHE Protocol

## CKKS configuration (HEonGPU backend)

| Parameter | Value | Why |
|---|---|---|
| Scheme | CKKS | best for fixed-point real arithmetic |
| Backend | **HEonGPU** (vendored) | CUDA, free-form chain, native Galois rotations |
| Ring dimension $N$ | $2^{16} = 65{,}536$ | $32{,}768$ slots after CKKS pack-2 |
| Scale | $2^{40}$ | enough precision for cubic invsqrt LN |
| Chain length (multiplicative depth) | **22** | 18 per-layer + buffer for residual + setup |
| Polynomial modulus pieces | 22 primes (~60 bit each) | matches chain |
| Bootstrap | **disabled** | one-pass forward fits in chain=22; bootstrap deferred |
| Hardware | single H100 SXM5 (80 GB HBM3) | |

Backend wrapper: [`fhe_thesis/encryption/heongpu_backend.py`](../fhe_thesis/encryption/heongpu_backend.py).
Vendored library: [`third_party/HEonGPU/`](../third_party/HEonGPU/) (commit pinned in
[`third_party/HEonGPU.commit`](../third_party/HEonGPU.commit)).

### Why chain = 22 specifically

| Chain | Outcome |
|---|---|
| ≤ 20 | LayerNorm cubic invsqrt polyval runs out of levels mid-circuit |
| 21 | works but no margin for the residual addition's depth alignment |
| **22** | **stable end-to-end forward pass for 12 layers** |
| 24 | also works (used in early BSGS run, BATCH=8); slower per-op |

Chain budget shopping: [`scripts/bench_L128_synthesizer_lpan.py`](../scripts/bench_L128_synthesizer_lpan.py)
sweeps `BATCH ∈ {4, 8, 16}` × `chain ∈ {21, 22, 24}` and reports
per-layer + total wall-time.

## Slot layout — NEXUS column-major

For each head ciphertext bundle:

```
slot[h·H + j·L + i] = X[h, i, j]       where H = head_dim · L
                                              h = head index in bundle
                                              i = token position
                                              j = embedding coordinate
```

Why column-major: makes the cyclic-shift along $i$ (token positions)
into a single Galois rotation, which is what every diagonal of the
Synthesizer pattern needs.

## Per-layer depth budget (chain = 22)

```
Layer ℓ uses:
  V projection (BSGS):                2 levels
  Synthesizer attention A·V:          1 level   ← single pt·ct
  Output projection:                  2 levels
  FFN linear 1:                       2 levels
  Polynomial GELU (deg 6):            3 levels
  FFN linear 2:                       2 levels
  Cubic-invsqrt LayerNorm:            6 levels  ← deepest single op
  ───────────────────────────────────────────
  Per-layer total:                   18 levels  (consumed)
```

After 12 layers we are far over chain=22 — but each layer **resets**
the chain by virtue of HEonGPU's `mod_drop_inplace_ct` + key-switching
bookkeeping. The 22-level budget is the **per-layer working budget**,
not a 12-layer cumulative.

## Token batching (BATCH=16)

Independent samples are packed in the slot dimension:

```
slot[batch_idx · (H·n_heads) + h·H + j·L + i]
```

Polynomial GELU and LayerNorm are slot-wise → amortize for free.
Per-sample throughput at BATCH=16:  60.9 s / 16 ≈ **3.8 s per sample**.

## Polynomial coefficient management

All coefficients are precomputed offline and shipped as plaintext
constants. There is no mid-circuit fitting.

| Activation | Approximation | Degree | Module |
|---|---|---|---|
| GELU | Chebyshev minimax on $[-4, 4]$ | 6 | [`fhe_thesis/poly/approximation.py`](../fhe_thesis/poly/approximation.py) |
| Softmax (legacy LPAN, ablation only) | Chebyshev minimax | 12 | same |
| LayerNorm $1/\sqrt{x}$ | cubic Newton step from initial guess | 3 | [`fhe_thesis/encryption/layernorm.py`](../fhe_thesis/encryption/layernorm.py) |

Empirical input ranges are tracked during plaintext fine-tuning via
hooks and frozen at export time — the deployed coefficients match the
deployed range.

## End-to-end protocol

```
Client                                                    Server
──────                                                    ──────
1. plaintext embed query x
2. encrypt → enc(x)                                       ──→
3.                                                        4. run 12-layer
                                                             Synthesizer-LPAN
                                                             forward in CKKS
                                                             (60.9 s on H100)
6. decrypt logits, argmax                            ←──   5. return enc(logits)
```

Single round, no MPC handshakes, no mid-circuit decryption.

## What we deliberately avoid

| Anti-pattern | Reason |
|---|---|
| Bootstrapping mid-forward | not needed at chain=22; complicates timing |
| Per-query attention pattern | breaks the Synthesizer architectural lever |
| Mid-circuit decryption | violates pure-FHE threat model |
| MPC fallback for non-polynomial ops | violates pure-FHE threat model |
| Branching on ciphertext content | impossible under CKKS by construction |
| Sending intermediate ciphertexts to client | breaks single-round non-interactivity |

See [03_THREAT_MODEL.md](03_THREAT_MODEL.md) for the full audit.
