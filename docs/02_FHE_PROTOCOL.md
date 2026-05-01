# 02 — FHE Protocol

## CKKS configuration

| Parameter | Value | Why |
|---|---|---|
| Scheme | CKKS (OpenFHE 1.2.3) | best for fixed-point real arithmetic |
| Multiplicative depth | 25 | covers per-layer max + bootstrap budget |
| Ring dimension N | 65 536 (2¹⁶) | 4096 plaintext slots after sparse packing |
| Used slots per ciphertext | up to 4096 | one ciphertext per token (token-packed) |
| Bootstrap | enabled, level budget [3, 3] | EvalBootstrap with HEXL acceleration |
| Number of threads | 32 (= 32 vCPU pod) | OpenFHE OMP parallel NTTs |

Backend: [`fhe_thesis/encryption/openfhe_backend.py`](../fhe_thesis/encryption/openfhe_backend.py).

## Token-packed layout

Each token is one ciphertext encrypting all `hidden_dim` coordinates
in adjacent slots. A sequence of `L` tokens is `L` ciphertexts in a
`TokenPackedTensor`. This trades higher latency on linear projections
(rotations + slot-wise muls) for trivial parallelism: every token can
be processed by an independent thread / vCPU.

Why token-packed (not feature-packed)?
- 32-vCPU pod parallelises naturally across tokens
- per-token isolation means word elimination (Ext W) is a free
  no-op — just drop ciphertexts, the schedule shrinks accordingly
- attention reduces to per-head dot-products and additions over the
  ciphertext list, no in-slot reductions needed

## Per-layer depth budget

See [`depth.py`](../fhe_thesis/encryption/depth.py) for symbolic costs
and [`bootstrap_scheduler.py`](../fhe_thesis/encryption/bootstrap_scheduler.py)
for the per-layer constants. In summary:

```
LinearMixing layer (LM):  attn=12 + ffn=11 = 23 levels
QuadAttention   layer (Q): attn=20 + ffn=11 = 31 levels
LPAN            layer (L): attn=22 + ffn=11 = 33 levels
```

Hot ops dominating each layer:
- `softmax_poly` (deg 12): 4 levels  ← LPAN only
- `quad_scores`: 2 levels             ← Quad only
- `pos_mix`: 1 level                  ← LinearMixing only
- `ln_poly` (with mean centring): 8 levels (largest single op)
- `polyval_deg6` GELU: 3 levels
- `qk_scores`: 2 levels
- `attn_apply`: 3 levels

## Bootstrap scheduling (Ext 1)

Standard practice bootstraps every K layers. We instead place refresh
points at the *latest* boundary that still fits the per-window budget,
saving bootstraps in compositions where early layers are LinearMixing.

API: [`fhe_thesis/encryption/bootstrap_scheduler.py`](../fhe_thesis/encryption/bootstrap_scheduler.py)

```python
from fhe_thesis.encryption.bootstrap_scheduler import (
    schedule_bootstraps, composition_to_kinds,
)
kinds = composition_to_kinds(12, lm_layers=[0,1,2,3], q_layers=[4,5,6,7])
plan  = schedule_bootstraps(kinds, budget_per_window=80)
print(plan.insertion_indices)  # [3, 5, 7, 9, 11] for budget=80
```

The protocol's `encrypt_inference_hybrid` accepts an optional
`bootstrap_plan` and calls `maybe_bootstrap(backend, x, plan, layer_idx)`
before each layer.

## Polynomial coefficient management

Two parallel paths share the same Chebyshev evaluator (Clenshaw recurrence):

1. **Fixed coefficients** (default): pre-fitted Remez/Chebyshev minimax
   on a worst-case interval, baked into the model at deployment.
   Loaders: `load_coefficients` / `load_coefficients_for_hybrid`.
2. **Learnable coefficients** (Ext 2): `LearnablePolyAdapter` exposes
   `coeffs` as `nn.Parameter` and tracks empirical input range via EMA.
   Frozen at inference and exported the same way as fixed coefficients.

Both paths produce the same on-disk format → the FHE backend doesn't
care which one was used.

## Word elimination protocol plumbing (Ext W)

Client-side, per query:

1. Tokenise → get `attention_mask` (and optionally a plaintext teacher
   model's layer-0 attention scores).
2. `apply_elimination(emb, mask, strategy=...)` returns
   `(filtered_emb, kept_indices)`.
3. Send `filtered_emb` (encrypted) + `kept_indices` (plaintext) to server.

Server-side, per layer:

```python
encrypt_layer_dispatch(..., kept_token_indices=kept_indices)
# → forwards to encrypt_linear_mixing_block, which slices
#   P[:, kept_idx, :][:, :, kept_idx] and biases[:, kept_idx]
```

Quad and LPAN layers are position-agnostic and operate directly on
the kept-token sequence — no plumbing needed for them.

## Decomposition of one sample's wall time

(Pod, 32 vCPU, max_seq_len=64, no Ext W; values from
`results/benchmarks/` and projected for unimplemented optimisations.)

| Phase | Time | Notes |
|---|---|---|
| Embedding (plaintext) | 0.05 s | client-side |
| Encrypt 64 tokens | 0.4 s | one ciphertext per token |
| 12 layers (avg LPAN-style) | 8–14 s | depends on composition |
| Bootstraps (3–5) | 2–4 s | EvalBootstrap with HEXL |
| Classifier head | 0.3 s | small linear |
| Decrypt CLS logits | 0.05 s | |
| **Total** | **~11–18 s** | shrinks to **5–9 s** with Ext W + Phase 2c |

See [06_HARDWARE.md](06_HARDWARE.md) for the full latency breakdown
and target.

## What we explicitly avoid (FHE-purity)

| Anti-pattern | Why we never do it |
|---|---|
| Decrypt mid-circuit | requires server to hold secret key — breaks threat model |
| MPC handshakes (BOLT, Iron) | pulls us out of pure FHE category, weaker security model |
| Trusted Execution Environments (Iron-LM, etc.) | architectural hardware trust assumption we don't want |
| Branch on ciphertext content | impossible in FHE; if a design needs it, redesign |
| Send intermediate ciphertexts back to client | also breaks single-round non-interactivity |

See [03_THREAT_MODEL.md](03_THREAT_MODEL.md) for the full audit table.
