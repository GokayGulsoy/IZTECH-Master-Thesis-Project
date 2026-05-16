# 00 — Project Overview

## Thesis title (working)

> **Synthesizer-LPAN: Sub-100-Second Single-GPU FHE BERT Inference via
> Architectural Elimination of Q, K and Softmax**

İzmir Institute of Technology (İYTE) — Master's thesis,
candidate: Gökay Gülsoy. Department of Computer Engineering.

## One-paragraph summary

We replace the standard self-attention block of BERT with a frozen,
*pre-softmaxed* learned attention pattern $A \in \mathbb{R}^{L\times L}$
(Tay et al., NeurIPS 2020 — "Synthesizer"). Under FHE this eliminates
the two query/key projections, the $Q\cdot K^\top$ ciphertext-by-ciphertext
multiplication, and the depth-12 softmax-poly approximation **entirely**.
On top of this we apply Baby-Step Giant-Step (BSGS) fusion of the
attention mask with each plaintext diagonal, batched LPAN evaluation,
and a careful CKKS chain budget. The result is a **single-GPU
sub-100-second end-to-end coherent FHE BERT result on plain HEonGPU**
at sequence length 128 — **60.9 s on one H100**, a 13.67× speedup over
our honest LPAN baseline of 833 s.

## Contributions

1. **First FHE port of Synthesizer attention.** Plaintext Synthesizer
   was discarded by the ML community in 2020 because it offered little
   speedup over standard MHA. Under FHE the entire $L^2$ ciphertext-by-
   ciphertext floor disappears — making this an architectural lever
   that LPAN-family papers (NEXUS, MPCFormer, BOLT, Iron) all left on
   the table.

2. **BSGS-fused mask × diagonal.** The naive Synthesizer attention
   needs $2L$ rotations for the V cyclic-shift plus $L$ plaintext
   multiplications per head bundle. By fusing the cyclic-shift mask
   with the per-diagonal pattern at encoding time and applying
   Halevi-Shoup BSGS, we cut rotations from $2L$ to $2 \cdot 2\sqrt{L}$
   — an 8× rotation reduction at $L=128$.

3. **Batched LPAN polynomial evaluation.** We process `BATCH=16`
   independent samples per ciphertext slot block. Polynomial
   activations (GELU, LayerNorm-invsqrt) amortize over the batch,
   dropping per-sample latency by ≈3.7×.

4. **Self-contained reproducible artifact.** HEonGPU is vendored
   (`third_party/HEonGPU/`, commit pinned). One `scripts/setup_pod_gpu.sh`
   on a stock Ubuntu + CUDA 12 + H100 machine reproduces the 60.9 s
   number end-to-end.

## Headline metric

```
12-layer Synthesizer-LPAN BERT, L=128, BATCH=16, chain=22, HEonGPU CKKS N=2^16
Single H100 SXM5 wall-time: 60.9 s
Honest LPAN baseline:        833 s
Speedup:                     13.67×
```

## Comparison axis (concurrent work)

CERIUM (arXiv:2512.11269, Dec 2025, CMU + NVIDIA) reports 8.8 s BERT-Base
on **8× B200** GPUs and 36.1 s on a single H100 — but as a *framework*
(DSL + compiler + runtime) running plain BERT. Our work is *architectural*:
we change BERT itself so the FHE circuit becomes 13.67× cheaper. The two
contributions compose; they are not competitors.

## Target venues

| Venue | Deadline | Track | Framing |
|---|---|---|---|
| **USENIX Security 2027** | Feb 2027 | full paper | primary; pure-FHE threat model |
| **EMNLP 2027** | Jun 2027 | long paper | secondary; ML/NLP framing |
| **ICLR 2027** | Sep 2026 | poster | early visibility for the architectural lever |

## Key validated numbers

| Metric | Value | Source |
|---|---|---|
| Synthesizer-LPAN end-to-end fwd (12L, L=128) | **60.9 s** | `scripts/bench_L128_synthesizer_lpan.py` on H100 |
| Honest LPAN baseline (12L, L=128) | 833 s | same harness, full softmax-poly |
| Speedup | **13.67×** | derived |
| Numerical correctness vs plaintext | tested per layer | `scripts/test_synthesizer_lpan_correctness.py` |
| Plaintext Synthesizer GLUE accuracy (Tay 2020) | > 97% of standard MHA | NeurIPS 2020 paper |

## Locked design decisions

- **CKKS backend**: HEonGPU (vendored), $N = 2^{16}$, scale $2^{40}$,
  chain length 22.
- **Sequence length**: $L = 128$.
- **Batch size**: 16 samples per slot block.
- **Attention pattern**: frozen, learned-once, plaintext (not data-dependent).
- **GELU / LayerNorm**: degree-6 / degree-3 (cubic invsqrt) Chebyshev minimax.
- **Hardware target**: single H100 SXM5. Multi-GPU explicitly out of scope
  (CERIUM owns that axis).

## Critical constraint

**Pure non-interactive FHE.** No mid-circuit decryption, no MPC
handshakes, no TEE. See [03_THREAT_MODEL.md](03_THREAT_MODEL.md).
