# LPAN-FHE Protocol Specification

> **Status:** Phases 1–3 implemented (model-agnostic for all `MODEL_REGISTRY` BERT variants).
> **Branch:** `feature/ckks-protocol`.

## 1. Motivation

Existing FHE/MPC transformer-inference systems (THE-X, MPCFormer, BOLT,
Iron, PEGASUS) all suffer from one common drawback: **they cannot
homomorphically evaluate LayerNorm**. Computing `(x − μ)/σ` requires a
square root and a division on a per-token statistic — operations whose
faithful FHE realisation would consume more multiplicative depth than
the rest of the layer combined. Every system therefore falls back to
**multi-party computation (MPC) round-trips**: the partial result is
sent back to the client (or to a non-colluding party), the variance is
computed in the clear, the inverse-square-root is applied, and the
re-encrypted tensor is sent back.

This is the dominant source of communication latency in those systems.
It also breaks the security model in inconvenient ways: the client now
has to be **online** for every layer.

LPAN's contribution makes a stronger protocol possible:

* GELU → degree-8 polynomial (Stage 1).
* Softmax → per-head degree-8 polynomial (Stage 2).
* **LayerNorm → polynomial (Stage 3).** ← this is the new ingredient.

Because every non-linear operation is now a low-degree polynomial, the
server can evaluate **the entire transformer layer purely under FHE**,
with no MPC round-trip. The client encrypts the input once, sends one
ciphertext, and receives one ciphertext back.

We call this the **Pure-FHE Single-Round (PF-SR) protocol**.

## 2. Threat Model

* Honest-but-curious server.
* Semi-honest client (provides correctly-encrypted inputs).
* Adversary: passive, computationally bounded, sees only ciphertexts.
* Security parameter: 128-bit classical, per HomomorphicEncryption.org
  CKKS standard.
* No assumption of a non-colluding third party (unlike Iron, MPCFormer).

## 3. Protocol

```
Client                           Server
──────                           ──────
1. Tokenise + embed(x)
2. ct_in ← Enc(pk, x)
                ──ct_in──▶
                                 3. ct_out ← LPAN_layer(ct_in)
                                    (no client interaction)
                ◀──ct_out──
4. y ← Dec(sk, ct_out)
5. argmax(y) → label
```

* **One round-trip total.** Independent of model depth.
* No partial decryptions, no garbled circuits, no OT.
* Client compute = embedding lookup + one Enc + one Dec.

For multi-layer models the server simply chains layers in-place; no
client interaction is ever required mid-inference.

## 4. Packing Strategy

### 4.1 Layout

We adopt **token-packed CKKS encoding**:

* Input tensor `X ∈ R^{seq_len × hidden_dim}` is encoded as
  `seq_len` separate ciphertexts, each holding one token's
  hidden-dim slots.
* Slot count per ciphertext = `N/2` (CKKS half-slot). For
  N = 16384 we have 8192 real slots, comfortably enough for
  hidden_dim ∈ {128, 256, 512, 768}.

```
seq_len = 128, hidden = 128
─────────────────────────────────
ct_0 = [x_{0,0}, x_{0,1}, …, x_{0,127}, 0, 0, …, 0]
ct_1 = [x_{1,0}, x_{1,1}, …, x_{1,127}, 0, 0, …, 0]
…
ct_127
```

### 4.2 Why this layout

LPAN polynomials (GELU, softmax-poly, LN-poly) are all **intra-token
element-wise** operations. Token-packing therefore evaluates them with
**zero rotations** — the most expensive CKKS primitive after
bootstrapping. Concretely:

| Op | Cost under token-packing |
|---|---|
| GELU(x) | 1 `polyval` per token ct, no rotations |
| LN-poly(x) | (Σx²) via log₂(hidden) rotations + 1 `invsqrt` polyval (per token only, **not per layer-wide**) |
| softmax-poly(row) | per-row, equals per-token here, no rotations |
| Linear `Wx + b` | matrix–vector via baby-step/giant-step over hidden_dim |
| Q · Kᵀ | inner-product per (i,j) pair → reuses linear primitive |

### 4.3 Why not the alternatives

| Layout | Pro | Why we don't use it |
|---|---|---|
| Hidden-packed (slots = seq_len) | row-wise softmax cheap | breaks LN-poly (LN is per-token, would need transpose) |
| Diagonal/Halevi-Shoup | best for square matmul | no benefit when hidden_dim ≪ N/2; rotations dominate polys |
| Full-batch row-major (BOLT) | one ct per layer | doesn't fit BERT-Base (128·768=98k > 8192) without splitting |

## 5. Multiplicative-Depth Budget

Per BERT layer, with degree-8 polynomials evaluated by Horner. Q, K, V
linears run on independent ciphertexts and so contribute **one** level
to the critical path, not three. Same for W₁ and W₂ (sequential, two
levels each):

| Stage | Critical-path depth |
|---|---|
| LN-poly (pre-attn) | 4 |
| Q-linear (∥ K, V) | 1 |
| Q · Kᵀ scores (dot + slot mask) | 2 |
| Softmax-poly | 3 |
| attn · V (mul_plain mask + ct·ct broadcast) | 2 |
| Head concat (mul_plain mask) | 1 |
| O-linear | 1 |
| residual + LN-poly | 4 |
| W₁ | 1 |
| GELU-poly | 3 |
| W₂ | 1 |
| residual | 0 |
| **Total per layer** | **23** |

The exact value is computed by `fhe_thesis.encryption.depth.transformer_layer_depth()`.

**Implications for backend parameters.** A 20-level critical path with
40-bit primes plus two 60-bit endpoints requires
60 + 20·40 + 60 = 920 bits of coefficient modulus. At 128-bit
classical security this fits N = 32768 (max ≈ 881 bits at HE-Standard)
**only after** one of:

* dropping the per-poly degree from 8 to 6 (saves 1 level per
  GELU/softmax/inv-sqrt block → 3 levels per layer),
* using 36-bit middle primes (saves ~80 bits),
* or admitting one bootstrap per layer.

For BERT-Tiny (2 layers, 40 levels) the answer will likely be the
36-bit prime variant; for BERT-Base (12 layers) bootstrapping cannot
be avoided. Phase-1 implementation uses N = 16384 with 6 mid-primes
which is sufficient for the FFN+LN block alone (depth = 9).

## 6. Implementation Roadmap

| Phase | Deliverable | Branch state |
|---|---|---|
| **1** ✅ | Design doc, backend abstraction, token-packed tensor, encrypted FFN+LN block | scaffold + Tiny FFN |
| **2** ✅ | Encrypted multi-head self-attention (Q/K/V projections per head, scaled-dot-product scores, softmax-poly, attn·V, zero-pad head concat) | adds `enc_self_attention` |
| **3** ✅ | Model-agnostic full layer + classifier head, unified CLI runner | adds `protocol.py`, `coefficients.py`, `experiments/run_protocol.py` |
| 4 | Scaling benchmark across Tiny / Mini / Small / Base | benchmark table |
| 5 | GPU-backend port (Phantom-FHE or OpenFHE-CUDA) | optional, perf-only |

## 7. Module Layout

```
fhe_thesis/encryption/
  __init__.py       # PEP-562 lazy imports (so design machine works without TenSEAL)
  context.py        # CKKS context factories
  backend.py        # CKKSBackend ABC + TenSEALBackend
  packing.py        # TokenPackedTensor
  ops.py            # enc_linear, enc_gelu_poly, enc_ln_poly,
                    # enc_qk_scores, enc_softmax_poly,
                    # enc_attention_apply, enc_self_attention
  coefficients.py   # PolyCoeffs + load_coefficients(model_key)
  protocol.py       # encrypt_ffn_block / encrypt_attention_block /
                    # encrypt_layer / encrypt_inference / run_phase
  depth.py          # symbolic depth tracker (DEPTH_COST, DepthAudit,
                    #   transformer_layer_depth() = 23)
experiments/
  run_protocol.py   # unified CLI: --model {tiny|mini|small|base}
                    #              --phase {ffn|attention|layer|model}
docs/
  ckks_protocol.md  # this document
```

## 8. Out of Scope (for Phases 1–2)

* Bootstrapping — we aim to avoid it entirely.
* GPU backend — will be added in Phase 5 if needed.
* Malicious-server security — honest-but-curious is enough for thesis.
* Encrypted embeddings — client owns the embedding table.

## 9. Phase 2 Notes — Encrypted Multi-Head Self-Attention

### 9.1 Per-head splitting under FHE

Each attention head is realised by **slicing the plaintext Q/K/V
weight matrices row-wise** into pieces of shape `(head_dim, hidden)`
and running `enc_linear` per head. The result is `num_heads`
independent `TokenPackedTensor`s, each of width `head_dim`. No
ciphertext is ever decrypted to materialise heads — the PF-SR
invariant is preserved.

### 9.2 Q·Kᵀ scores layout

For a query row `i`, scores against all keys are packed into a single
ciphertext of `seq_len` slots:

```
for j in 0..L-1:
    s_j        = backend.dot(Q[i], K[j])         # broadcast scalar
    masked     = mul_plain(s_j, one_hot(j) * 1/√d)
    row_ct    += masked
```

The `1/√d` scale is folded into the mask so the softmax polynomial
sees scores in its profiled interval.

### 9.3 attn·V layout

For each output row `i`:

```
for j in 0..L-1:
    a_ij_scalar = sum_slots(mul_plain(attn[i], one_hot(j)))
    out_i      += V[j] * a_ij_scalar     # ct·ct broadcast
```

Cost per row: `seq_len × (mul_plain + sum_slots + ct·ct + add)`.

### 9.4 Head concatenation without decryption

Each head's output occupies its `head_dim` slots of every token ct
(rest are zero by encoding). To concatenate, multiply each head's
ciphertext by a per-head plaintext mask that is `1` only inside that
head's destination slot range, then add. The cost is `num_heads ×
seq_len × (mul_plain + add)` and consumes one extra level.
