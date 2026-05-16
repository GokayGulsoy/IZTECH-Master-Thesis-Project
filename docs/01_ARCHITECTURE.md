# 01 — Synthesizer-LPAN Architecture

> **Synthesizer-LPAN** = **Synthesizer Learnable Polynomial Activation Network**

A 12-layer BERT-style encoder where:

- **Self-attention** is replaced by **Synthesizer attention** (Tay et al.,
  NeurIPS 2020): a frozen, learned, **plaintext** attention pattern
  $A \in \mathbb{R}^{L \times L}$ per head with the softmax already
  absorbed at training time.
- **GELU**, **softmax** (in the rare layers that keep it for ablation),
  and **LayerNorm** are replaced with **learnable polynomial activations
  (LPAN)** — Chebyshev minimax with empirically tracked input ranges.

## What Synthesizer eliminates under FHE

Standard self-attention needs:

| Op | Cost in FHE | Standard MHA | Synthesizer |
|---|---|---|---|
| Wq projection | linear, $L \cdot d^2$ pt·ct | required | **eliminated** |
| Wk projection | linear, $L \cdot d^2$ pt·ct | required | **eliminated** |
| $Q \cdot K^\top$ | $L^2$ ciphertext × ciphertext | required | **eliminated** |
| softmax-poly | depth-12 Chebyshev (4 levels) | required | **eliminated (absorbed in $A$)** |
| $A \cdot V$ | depends on layout | $L^2$ ct·ct | **$L$ pt·ct** |
| Wv projection | linear | required | required |
| Output proj. | linear | required | required |

Plaintext Synthesizer trades a small accuracy loss for negligible
speedup → discarded by ML community in 2020. Under FHE the entire
$L^2$ ct·ct floor disappears → the architectural lever that all
LPAN-family papers missed.

## Mathematical contract

For each head $h$ in each layer $\ell$:

$$
O_{h} = A_h \cdot V_h, \qquad A_h \in \mathbb{R}^{L\times L} \text{ (plaintext, frozen)}, \qquad V_h \text{ (ciphertext)}.
$$

$A_h$ is fitted **once at training time** by distillation from a
plaintext fine-tuned BERT. At inference time $A_h$ is encoded as a
plaintext and reused for every query → zero per-inference encoding
cost.

## Per-layer depth (CKKS levels, HEonGPU N=2¹⁶)

| Component | Levels |
|---|---|
| V projection (BSGS) | 2 |
| Synthesizer attention $A \cdot V$ | 1 (single pt·ct multiply) |
| Output projection | 2 |
| FFN linear 1 | 2 |
| Polynomial GELU (degree 6) | 3 |
| FFN linear 2 | 2 |
| LayerNorm (cubic invsqrt + mean centring) | 6 |
| **Per-layer total** | **18** |

Critical-path observation: LayerNorm's cubic invsqrt is the
single deepest op (6 levels) — this is why chain budgets below 21
fail (21 = LN + buffer for residual + bootstrap setup overhead).
Final tuned chain = **22**.

## Algorithm — diagonal decomposition with BSGS fusion

V is laid out in NEXUS column-major order: `slot[h·H + j·L + i] = V[i, j]`,
where `H = head_dim · L`. We compute $O = A \cdot V$ by summing
diagonals:

```
For each diagonal d ∈ [0, L):
  V_shift_d  = cyclic_shift_along_i(V, d)    # 2 rotations + 2 mul_plain
  diag_mask_d = encode(A[h, i, (i+d) mod L]) # plaintext, encoded ONCE
  O += V_shift_d ⊙ diag_mask_d                # 1 mul_plain + 1 add
```

### Naive cost

| Op | Count |
|---|---|
| ciphertext rotations | $2L$ |
| pt·ct multiplications | $3L$ |
| ciphertext additions | $3L$ |
| ct·ct multiplications | **0** |

vs standard MHA which needs $L$ ct·ct (depth-2) multiplications for
$Q\cdot K^\top$ alone — already a >10× depth-weighted reduction.

### BSGS-fused improvement

Insight: the cyclic-shift mask `mask_top[d]` and the diagonal pattern
`diag[d]` are both plaintext. Fuse them at encoding time:

```
top_combined[d] = mask_top[d] ⊙ diag[d]   (pre-encoded plaintext)
bot_combined[d] = mask_bot[d] ⊙ diag[d]   (pre-encoded plaintext)
```

Then apply Halevi-Shoup BSGS with $bs \cdot gs = L$, both $\approx \sqrt{L}$:

```
For g ∈ [0, gs):
  V_giant_g = rotate(V, g·bs)                       # 1 rotation per g
  For b ∈ [0, bs):
    O += V_giant_g ⊙ top_combined_pre_rotated[g][b] # 1 pt·ct + 1 add
For g ∈ [0, gs):  # bot mirror
  ... (same structure)
```

Final cost (BATCH=16, L=128):

| Op | Naive | BSGS-fused | Reduction |
|---|---|---|---|
| rotations | 256 | **32** | **8×** |
| pt·ct | 384 | 256 | 1.5× |
| add | 384 | 256 | 1.5× |
| ct·ct | 0 | 0 | — |

Wall-time impact: 222.8 s (naive) → 60.9 s (BSGS) at BATCH=16.

## Batched LPAN evaluation

Polynomial activations (GELU, LayerNorm invsqrt) are slot-wise
operations on ciphertexts. Packing `BATCH=16` independent samples
into the slot dimension means one polynomial-evaluation circuit
amortises across 16 inferences:

```
Per-sample latency = (12 layers × per-layer cost) / BATCH
                   = (12 × 81 s) / 16  ≈ 60.9 s   total wall-time per batch
                   ≈ 3.8 s per sample (effective throughput)
```

The CKKS slot count $N/2 = 32768$ comfortably hosts
`BATCH × head_dim × L = 16 × 64 × 128 = 131072` only when split
across multiple ciphertexts — `multi.py` orchestrates this.

## Frozen attention pattern $A$

$A$ is fitted offline:

1. Train standard BERT with full self-attention on the target task
   (e.g., SST-2).
2. For each layer / head, run inference on the training set with
   `output_attentions=True`.
3. Average the attention matrices over the training set per head.
4. Apply small per-task fine-tuning of $A$ as a free
   `nn.Parameter` (plaintext) with the rest of the model frozen.
5. Quantize / clip $A$ to fit the CKKS scaling budget.

After this, $A$ is **a plaintext constant** of the deployed model —
identical security treatment as model weights.

## What we explicitly do not do

| Anti-pattern | Why we avoid it |
|---|---|
| Per-query attention | would require $Q, K$ → defeats the architectural lever |
| Linformer-style sequence projection | NEXUS column-major makes sequence-mixing rotations expensive (only 1.2× speedup measured) |
| Multi-GPU sharding | out of scope; CERIUM owns that axis |
| Mixed-precision FHE (CKKS + TFHE) | adds protocol complexity; defers to future work |
| Trainable polynomial in inference | coefficients frozen at deploy — same security as weights |

## Code map

| Concern | Module |
|---|---|
| Synthesizer attention (naive + BSGS) | [`fhe_thesis/encryption/attention.py`](../fhe_thesis/encryption/attention.py) |
| Linear projections (BSGS, streaming, multi) | [`fhe_thesis/encryption/linear.py`](../fhe_thesis/encryption/linear.py) |
| LayerNorm (cubic invsqrt + mean centre) | [`fhe_thesis/encryption/layernorm.py`](../fhe_thesis/encryption/layernorm.py) |
| Column-major packing helpers | [`fhe_thesis/encryption/colmajor.py`](../fhe_thesis/encryption/colmajor.py) |
| Multi-ciphertext bundles | [`fhe_thesis/encryption/multi.py`](../fhe_thesis/encryption/multi.py) |
| HEonGPU backend (CUDA) | [`fhe_thesis/encryption/heongpu_backend.py`](../fhe_thesis/encryption/heongpu_backend.py) |
| Polynomial Chebyshev fits | [`fhe_thesis/poly/approximation.py`](../fhe_thesis/poly/approximation.py) |
| Polynomial activation modules (PyTorch) | [`fhe_thesis/models/activations.py`](../fhe_thesis/models/activations.py) |
| Surgery: replace BERT activations | [`fhe_thesis/models/replacement.py`](../fhe_thesis/models/replacement.py) |
