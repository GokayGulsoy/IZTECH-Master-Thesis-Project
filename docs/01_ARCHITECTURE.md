# 01 — HyPER-LPAN Architecture

**HyPER-LPAN = Hybrid Per-region Encrypted Reasoning with LPAN.**

A 12-layer BERT-style encoder where each layer's attention block is one
of three primitives, picked per-layer and per-task:

| Primitive | Cost (LM-relative) | What it can do | What it can't |
|---|---|---|---|
| **LinearMixing (LM)** | 1.0× | learned position mixing (FNet-like) | content-aware attention |
| **QuadAttention (Q / 2Quad)** | 1.4× | content-aware attention with squared scores (no softmax) | sharp attention peaks |
| **LPAN (L)** | 3.5× | full softmax-poly attention (Cheby approximation) | nothing — most expressive |

LinearMixing has zero ct×ct multiplications; Q has 2; LPAN has ~5 per
attention block. FFN block (linear + poly-GELU + linear + poly-LN) is
identical across all three.

## Per-region depth (CKKS levels)

| Layer kind | Critical-path depth |
|---|---|
| LinearMixing | 23 |
| QuadAttention | 31 |
| LPAN | 33 |

Depths are summed in [`fhe_thesis/encryption/depth.py`](../fhe_thesis/encryption/depth.py).

## Canonical composition (validated baseline)

```
L0  L1  L2  L3   L4  L5  L6  L7   L8  L9  L10 L11
─── LinearMixing ─── QuadAttention ─── LPAN ──────
```

Total summed depth: 4·23 + 4·31 + 4·33 = 348 levels.
Picked because *early* layers do diffuse mixing (don't need content),
*middle* layers do feature combination (need light content awareness),
*late* layers do prediction (need sharp attention).

This worked for SST-2 (90.83 %) but failed for MRPC (82.60 %) — see
the task-adaptive selector below.

## Task-adaptive composition (Ext 3 — headline contribution)

For each task, run the *plaintext* fine-tuned BERT on a small dev split
with `output_attentions=True` and measure per-layer attention entropy
(normalised by `log(seq_len)`):

$$
H_l = \mathbb{E}_{s,h,q}\!\left[ -\sum_k a_{l,h,q,k} \log a_{l,h,q,k} \right] / \log L
$$

- $H_l \in [0,1]$
- High $H_l$ → diffuse → use LinearMixing
- Low $H_l$ → peaked → must use LPAN
- Middle → QuadAttention

Two-threshold rule with thresholds chosen by sweeping the empirical
distribution under a per-task latency budget. Falls back to all-Quad if
no plan fits the budget.

Code: [`fhe_thesis/optimization/composition_selector.py`](../fhe_thesis/optimization/composition_selector.py)
CLI: [`experiments/select_composition.py`](../experiments/select_composition.py)

Example output (BERT-base + SST-2, budget = 0.5 × full-LPAN cost):

```
linear_mixing_layers: [0, 1, 2, 3, 4, 6, 8, 9, 10, 11]
quad_attention_layers: []
lpan_layers: [5, 7]
estimated_cost: 17.00 (vs full-LPAN 42.0, speedup 2.47×)
```

## Training pipeline (4 stages)

`experiments/train_hyper_lpan.py` runs all four in one resumable
invocation, driven by `configs/hyper_lpan/<task>_<model>.yaml`.

1. **Stage A — LPAN baseline**: replace softmax/GELU/LN with
   degree-12/6/4 Chebyshev approximations; KD-distill from FP32
   teacher. Output: `lpan_staged_final/`.
2. **Stage B — QuadAttention replacement** of the layers in
   `quad_attention_layers`. KD from the LPAN teacher (warm-start each
   replaced layer from its LPAN attention, shrink mid-training).
3. **Stage C — LinearMixing replacement** of the layers in
   `linear_mixing_layers`. Same KD recipe.
4. **Stage D — Global fine-tune**: unfreeze everything and fine-tune
   end-to-end with a small KD γ.

Each stage writes a checkpoint under `results/multi_model/<task>/
<model>/<stage_name>/best_model/` and a `.done` marker so the next
invocation skips it.

## Polynomial activations (LPAN)

| Activation | Degree | Domain | Module |
|---|---|---|---|
| GELU | 6 | learned per-layer (typically [-5, 5]) | `PolynomialGELU` |
| Softmax (per-head exp + normaliser) | 12 | shifted scores ≤ 0, [-15, 0.5] | `PerHeadPolynomialSoftmax` |
| LayerNorm inv-sqrt | 4 | [0.01, 50] | `PolynomialLayerNorm` |

Coefficients fit by Chebyshev-Remez minimax + range-tracking; see
[`fhe_thesis/poly/`](../fhe_thesis/poly/). Now also trainable via
`LearnablePolyAdapter` (Ext 2) — see
[04_EXTENSIONS.md](04_EXTENSIONS.md#ext-2--learnable-poly-coefficients).

## QuadAttention (2Quad) detail

Replaces softmax with squaring + scalar `1/L` normalisation:

```
scores = Q @ Kᵀ                    # 1 ct×ct (multi-head packed)
attn   = (scores)² / L             # 1 ct×ct + 1 mul-plain
out    = attn @ V                   # 1 ct×ct
```

Saves 2 levels per layer vs LPAN softmax. Module:
[`fhe_thesis/models/quad_attention.py`](../fhe_thesis/models/quad_attention.py).

## LinearMixing detail

```
y = P · x   where  P ∈ ℝ^{H × L × L}   (per-head, learned, position-only)
out = Wo · concat_heads(y)
```

Zero ct×ct multiplications. The position-mix matrix `P_h` is a
plaintext weight, evaluated via `mul_plain + add` per row. Module:
[`fhe_thesis/models/linear_mixing.py`](../fhe_thesis/models/linear_mixing.py).

When word elimination is active (Ext W), the protocol slices
`P[:, kept_idx, :][:, :, kept_idx]` so the per-head mixing matches the
surviving token positions. See
[`fhe_thesis/encryption/protocol.py`](../fhe_thesis/encryption/protocol.py)
function `encrypt_linear_mixing_block`.

## Backbone abstraction (Ext 4)

[`fhe_thesis/models/backbone.py`](../fhe_thesis/models/backbone.py)
exposes `get_backbone(model)`, `get_encoder_layers(model)`,
`get_embeddings(model)`, `num_layers(model)` so all transforms work on
BERT, RoBERTa, and DistilBERT without code changes. New architectures
are a one-line table addition.
