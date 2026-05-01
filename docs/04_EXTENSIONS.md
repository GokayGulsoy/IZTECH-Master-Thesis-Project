# 04 — The Five Extensions

All five live on `feature/hyper-lpan-extensions` (commits `d288662`
through `4bdb02e`, branched from validated baseline `14a43c2` on
`feature/ckks-protocol`).

| # | Name | Type | Headline benefit | Commit | Module |
|---|---|---|---|---|---|
| W | Word elimination | systems | -50 % linear ops, -75 % quadratic ops | `d288662` | `encryption/elimination.py` |
| 3 | Task-adaptive composition | ML | per-task latency/accuracy Pareto | `71c25b9` | `optimization/composition_selector.py` |
| 1 | Region-adaptive bootstrap | crypto | fewer bootstraps for cheap-region compositions | `6d9b249` | `encryption/bootstrap_scheduler.py` |
| 2 | Learnable poly + range tracking | ML | tighter approximation → +0.2–0.4 acc | `c9cd28d` | `poly/learnable.py` |
| 4 | RoBERTa + DistilBERT support | generality | 3 backbones from 1 codebase | `4bdb02e` | `models/backbone.py` |

---

## Ext W — Word Elimination

**Inspired by**: BOLT-W.E. (S&P'24, oblivious bitonic sort over MPC).
**Adapted as**: pure FHE — client-side keep-decision, server slices
position-mixing matrices.

### Two strategies

#### `padding` (default, free, lossless)

Drop `[PAD]` tokens client-side using the standard `attention_mask`.
Lossless because polynomial-softmax attention has no mask anyway —
PAD tokens contribute spurious dot products that we'd rather not pay
for. Same leakage profile as standard FHE work (ciphertext count).

#### `content_teacher`

Client runs a small plaintext teacher (the FP32 BERT itself), takes
layer-0 attention CLS-row averaged over heads, keeps top-`keep_ratio`
tokens (CLS always force-kept). Sends kept indices in plaintext.

Speedup: roughly `1/keep_ratio` on linear ops, `1/keep_ratio²` on
quadratic ops. At `keep_ratio=0.5`: **2× linear, 4× quadratic**.

Extra leakage: the kept-token mask (positions, not values).

### API

```python
from fhe_thesis.encryption.elimination import (
    apply_elimination, elimination_savings,
)
filtered_emb, kept_idx = apply_elimination(
    emb, attention_mask,
    strategy="content_teacher", teacher_scores=scores, keep_ratio=0.5,
)
sav = elimination_savings(orig_seq_len=64, kept=len(kept_idx))
```

### Protocol plumbing

[`fhe_thesis/encryption/protocol.py`](../fhe_thesis/encryption/protocol.py)
functions `encrypt_linear_mixing_block`, `encrypt_layer_linear_mix`,
`encrypt_layer_dispatch`, `encrypt_inference_hybrid`,
`encrypt_inference_linear_mixing` all accept an optional
`kept_token_indices: np.ndarray`. LinearMixing layers use it to slice
`P[:, idx, :][:, :, idx]`. Quad/LPAN ignore it (position-agnostic).

CLI: `python experiments/run_fhe_benchmark.py --word-elimination padding|content_teacher [--keep-ratio 0.5]`

---

## Ext 3 — Task-Adaptive Composition Selector (HEADLINE)

### Why

The canonical [LM 0-3 + Q 4-7 + L 8-11] composition was *guessed*. It
worked for SST-2 (90.83 %) but failed catastrophically on MRPC
(82.60 % vs SOTA 88+). Different tasks need different compositions.

### Method

Plaintext fine-tuned model → run on small dev split with
`output_attentions=True` → measure normalised entropy
$H_l \in [0,1]$ per layer. Two-threshold rule:

```
H_l ≥ τ_high → LinearMixing   (cheapest, blind to content)
H_l ≤ τ_low  → LPAN           (most expensive, content-critical)
otherwise    → QuadAttention  (medium)
```

Budget mode: sweep candidate threshold pairs from the empirical
distribution, pick the one whose induced cost (LM=1.0, Q=1.4, L=3.5)
fits the budget while maximising LM use.

### API

```python
from fhe_thesis.optimization.composition_selector import compose_for_task
plan = compose_for_task(model, dev_samples, budget=21.0, min_lpan=2)
print(plan.linear_mixing_layers, plan.quad_attention_layers, plan.lpan_layers)
```

CLI: `python experiments/select_composition.py --model base --task mrpc --checkpoint <path>`

### Validation

BERT-base + SST-2, 16 dev samples, budget = 0.5 × full-LPAN cost:

```
linear_mixing_layers: [0,1,2,3,4,6,8,9,10,11]
quad_attention_layers: []
lpan_layers: [5, 7]
estimated_cost: 17.00 (vs full-LPAN 42.0, speedup 2.47×)
```

---

## Ext 1 — Region-Adaptive Bootstrap Scheduling

### Why

Standard CKKS deployments bootstrap every K layers. With heterogeneous
per-layer depth (LM=23, Q=31, L=33 levels), this wastes budget on
cheap-region transitions. We greedily place refresh points at the
*latest* layer boundary that still fits the per-window budget.

### API

```python
from fhe_thesis.encryption.bootstrap_scheduler import (
    schedule_bootstraps, composition_to_kinds, compare_plans,
)
kinds = composition_to_kinds(12, lm_layers=[0,1,2,3], q_layers=[4,5,6,7])
plan  = schedule_bootstraps(kinds, budget_per_window=80)
# plan.insertion_indices = [3, 5, 7, 9, 11]   ← 5 bootstraps
```

`compare_plans(kinds, B, uniform_period=2)` returns side-by-side
adaptive-vs-uniform comparison + bootstraps_saved count.

### Protocol plumbing

`encrypt_inference_hybrid(..., bootstrap_plan=plan)` calls
`maybe_bootstrap()` before each layer dispatch and records timing as
`L{i}.bootstrap`.

### Savings depend on composition + budget

For canonical LM+Q+L composition with budget=80 levels, adaptive ties
uniform (5 bootstraps each). Wins emerge for compositions with long
LM/Q prefixes — e.g. all-LM-then-all-LPAN saves 3+ bootstraps.

---

## Ext 2 — Learnable Polynomial Coefficients + Range Tracking

### Why

Fixed Chebyshev approximations are L∞-optimal *over a worst-case
range* that is much wider than the empirical activation distribution.
We waste approximation budget on tails the model never visits.

### Method

`LearnablePolyAdapter` wraps a Chebyshev series with:

1. `coeffs : nn.Parameter` (initialised from a fixed Chebyshev fit) —
   gradients flow through Clenshaw recurrence.
2. `range_min`, `range_max` : EMA-updated buffers tracking the
   empirical min/max during training.
3. `fidelity_loss()` : a regulariser
   $\frac{1}{N}\sum_i (p(x_i) - f(x_i))^2$ over Chebyshev nodes that
   bakes the target activation back in to prevent the model from
   "cheating" by overfitting an arbitrary polynomial.

Add `λ · collect_fidelity_loss(model)` to the distillation loss.

### API

```python
from fhe_thesis.poly.learnable import (
    LearnablePolyAdapter, collect_fidelity_loss, export_adapters_state,
)
ada = LearnablePolyAdapter(
    init_coeffs, target_fn=F.gelu, init_range=(-3, 3), name="L7.GELU",
)
# In training step:
loss = ce_loss + kd_loss + 0.01 * collect_fidelity_loss(model)
# At deploy:
export_adapters_state(model)  # {name: {coeffs, range_min, range_max}}
```

### Validation

50 Adam steps on a degree-6 GELU adapter drives fidelity loss to 1.3e-3
and matches `F.gelu` on `[-3, 3]` to within ~0.05 absolute error.

### FHE compliance

At deployment the trained coefficients and frozen empirical range are
baked into the published model — identical to fixed Chebyshev from a
security standpoint.

---

## Ext 4 — Cross-Architecture Backbone Resolver

### Why

Every replacement transform walked `model.bert.encoder.layer`
explicitly, hard-coding BERT. Adding RoBERTa/DistilBERT meant
duplicating four files.

### Method

`fhe_thesis/models/backbone.py` table:

```python
_BACKBONE_PATHS = (
    ("bert",       ("encoder", "layer")),
    ("roberta",    ("encoder", "layer")),
    ("distilbert", ("transformer", "layer")),
)
```

API: `get_backbone(model)`, `get_encoder_layers(model)`,
`get_embeddings(model)`, `num_layers(model)`. All four replacement
modules now use these helpers; adding a new HF architecture is a
one-line table addition.

### MODEL_REGISTRY additions

```yaml
roberta-base:  layers 12, hidden 768, heads 12, params 125M
distilbert:    layers  6, hidden 768, heads 12, params  66M
```

### Validation

BERT-base verified end-to-end. RoBERTa/DistilBERT require an HF cache
download and will be exercised in the final cross-architecture sweep.
