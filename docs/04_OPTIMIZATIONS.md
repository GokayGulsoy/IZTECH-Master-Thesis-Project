# 04 — Optimizations

The optimizations applied on top of the architectural baseline
(Synthesizer attention + LPAN polynomials), in chronological order
of implementation. Each row is a step we measured.

## Wall-time ladder (12-layer fwd, L=128, single H100)

| Step | Configuration | Wall-time | vs baseline |
|---|---|---|---|
| 0 | Honest LPAN baseline (full softmax-poly all 12 layers) | 833 s | 1.00× |
| 1 | Batched-shared-W LPAN, BATCH=4 | 534 s | 1.56× |
| 2 | Linformer-LPAN attempt (sequence projection) | 440 s (projected) | 1.89× |
| 3 | **Synthesizer-LPAN naive**, BATCH=4 | 222.8 s | 3.74× |
| 4 | Synthesizer-LPAN naive, BATCH=16 | 131.4 s | 6.34× |
| 5 | Synthesizer-LPAN BSGS, BATCH=8, chain=24 | 86.1 s | 9.67× |
| 6 | **Synthesizer-LPAN BSGS, BATCH=16, chain=22** | **60.9 s** | **13.67×** |

See [TECHNIQUES_JOURNEY.md](TECHNIQUES_JOURNEY.md) for the narrative
of *why* each step was tried and why some were abandoned.

---

## Optimization 1 — Synthesizer attention (architectural)

The single largest win (3.74×). Replaces $Q\cdot K^\top$, softmax-poly,
and the Wq/Wk projections with a frozen plaintext $A \in \mathbb{R}^{L \times L}$.
See [01_ARCHITECTURE.md](01_ARCHITECTURE.md) for math and depth math.

Code: [`encode_synthesizer_diagonals`](../fhe_thesis/encryption/attention.py),
[`attn_synthesizer`](../fhe_thesis/encryption/attention.py).

---

## Optimization 2 — Batched LPAN polynomial evaluation

Pack `BATCH=16` independent samples into the slot dimension. The
slot-wise polynomial activations (GELU, LayerNorm cubic invsqrt) then
amortize over the batch for free.

Per-sample effective latency at BATCH=16:  60.9 / 16 ≈ **3.8 s/sample**.

Limit: total slots used per ciphertext bundle =
`BATCH × num_heads × head_dim × L`. For BERT-base (12 heads, head_dim=64,
L=128), BATCH=16 fits across multiple ciphertext bundles orchestrated
by [`fhe_thesis/encryption/multi.py`](../fhe_thesis/encryption/multi.py).

---

## Optimization 3 — BSGS-fused mask × diagonal

The naive Synthesizer attention needs $2L$ rotations to produce the
$L$ cyclic shifts of $V$. By:

1. Pre-multiplying each cyclic-shift mask with its corresponding
   diagonal pattern at **encoding time** (both are plaintext).
2. Applying Halevi-Shoup BSGS with $bs \cdot gs = L$ both $\approx \sqrt{L}$.

we cut rotations from $2L$ to $2 \cdot 2\sqrt{L}$ — at $L=128$, from
**256 → 32**, an 8× reduction. Wall-time drops 131.4 s → 60.9 s.

Mathematical equivalence is exact (no approximation introduced).

Code: [`encode_synthesizer_bsgs`](../fhe_thesis/encryption/attention.py),
[`attn_synthesizer_bsgs`](../fhe_thesis/encryption/attention.py).

---

## Optimization 4 — CKKS chain tuning

Per-layer working depth = 18 levels. Practical chain budget choices:

| Chain | Stable? | Per-op cost | Wall-time |
|---|---|---|---|
| 21 | ❌ — LN runs out of levels | — | — |
| **22** | ✅ stable | smallest stable | **60.9 s** |
| 24 | ✅ stable | larger primes | 86.1 s |

Chain=22 hits the sweet spot: just enough headroom for LN's cubic
invsqrt while keeping per-op cost minimal. BATCH=16 only fits at
chain ≤ 22 due to slot pressure.

---

## Optimization 5 — Diagonal cache reuse across layers

The cyclic-shift `top_pts` / `bot_pts` masks depend only on $L$,
`head_dim`, `num_heads_per_ct` — not on layer-specific weights. We
cache them at first call:

```python
cache_key = ("synth_av_shift", L, head_dim, num_heads_per_ct)
```

→ 12-layer encode time amortizes over the first layer. Saves ~10 s
of plaintext-encoding setup per fresh inference.

---

## Optimization 6 — Vendored HEonGPU + commit pin

Reproducibility, not raw speed:

```
third_party/HEonGPU/        ← vendored 8 MB
third_party/HEonGPU.commit  ← pinned upstream commit
scripts/setup_pod_gpu.sh    ← one-shot build on stock Ubuntu + CUDA 12
```

Future HEonGPU versions cannot regress our numbers.

---

## Cancelled / failed optimizations

| Attempt | Result | Why cancelled |
|---|---|---|
| Linformer-LPAN sequence projection | only 1.2× projected | NEXUS column-major makes sequence-mixing rotations expensive |
| Plaintext softmax injection | breaks pure-FHE threat model | violates [03_THREAT_MODEL.md](03_THREAT_MODEL.md) |
| Bootstrap mid-forward | added ~30 s with no qualitative gain | chain=22 is sufficient |
| Multi-GPU sharding | out of scope | CERIUM owns that axis |
| Mixed CKKS + TFHE for nonlinearities | protocol complexity ↑ | deferred to future work |

Full narrative in [TECHNIQUES_JOURNEY.md](TECHNIQUES_JOURNEY.md).

---

## How to add a new optimization

1. Implement in `fhe_thesis/encryption/` against the `CKKSBackend`
   interface (so it works on both HEonGPU and any future backend).
2. Verify exact equivalence to the naive version on a small case via
   [`scripts/test_synthesizer_lpan_correctness.py`](../scripts/test_synthesizer_lpan_correctness.py).
3. Benchmark vs the wall-time ladder via
   [`scripts/bench_L128_synthesizer_lpan.py`](../scripts/bench_L128_synthesizer_lpan.py).
4. Update this doc + [TECHNIQUES_JOURNEY.md](TECHNIQUES_JOURNEY.md)
   with a row in the wall-time ladder.
