# 03 — Threat Model

## Setting

Single-server semi-honest FHE inference, single-round, non-interactive:

```
client                                                     server
─────                                                     ──────
1. holds secret key sk
2. encrypts query x → enc(x)
3. sends enc(x) + plaintext metadata (kept_indices, etc.)  →
                                                            4. runs the
                                                               full BERT
                                                               forward
                                                               under FHE
                                                            5. returns
   ←  enc(logits)                                              enc(logits)
6. decrypts logits, takes argmax
```

The server **never** sees plaintext, **never** decrypts anything,
**never** sends ciphertexts back to the client mid-inference.

## Adversary capabilities (semi-honest)

The server:
- follows the protocol honestly
- but tries to learn as much as possible from what it sees
- has unbounded compute and storage
- knows the public model (architecture, weights, polynomial coefficients,
  composition plan, bootstrap schedule)
- knows CKKS public parameters (ring dim, depth, scaling factor)

## Acceptable leakage (standard for ALL FHE work)

| Leakage | Why it's acceptable | Comparable to |
|---|---|---|
| Number of ciphertexts in the query | sequence length is needed by every protocol | BOLT, Iron, NEXUS, MPCFormer |
| Circuit shape (op counts, depth) | model is public | universal |
| Wall-clock timing of operations | side channel; addressed via constant-time at backend layer | |
| For Ext W content_teacher: the kept-token mask | reveals *positions* of important tokens, not values | new — discussed in paper §6 |

## What we do NOT leak

- Plaintext token values (`input_ids`)
- Embedding values
- Intermediate activations
- Attention weights
- Final logit values (only the encrypted logits leave the server)
- Predicted class label (client decides)

## Comparison vs related work

| System | Crypto | Server holds sk? | Decrypts mid-circuit? | MPC rounds | Threat model |
|---|---|---|---|---|---|
| **HyPER-LPAN (us)** | CKKS | ❌ | ❌ | 0 | strongest |
| NEXUS (Crypto'24) | CKKS | ❌ | ❌ | 0 | same as us |
| BOLT (S&P'24) | CKKS + 2PC | ❌ | yes (via SS) | many | weaker |
| Iron (NeurIPS'22) | CKKS + 2PC | ❌ | yes | many | weaker |
| Iron-LM (USENIX'24) | CKKS + TEE | ❌ | yes (in TEE) | 0 | TEE-trusted |
| MPCFormer (ICLR'23) | secret sharing | distributed | yes | many | weaker |
| THE-X (ACL'22) | CKKS + client compute | ❌ | sends partial back | 1 | weaker |

**Our threat model dominates BOLT/Iron/Iron-LM/MPCFormer/THE-X**.
Only NEXUS matches it. This is itself a publishable framing.

## Per-extension FHE-compliance audit

| Extension | Module | Mid-circuit decrypt? | MPC handshake? | TEE? | Branches on ct? | Verdict |
|---|---|---|---|---|---|---|
| HyPER-LPAN core | `protocol.py` | ❌ | ❌ | ❌ | ❌ | ✅ pure |
| Ext W (padding) | `elimination.py` | ❌ — uses public mask | ❌ | ❌ | ❌ | ✅ pure |
| Ext W (content_teacher) | `elimination.py` | ❌ — client teacher is plaintext | ❌ | ❌ | ❌ | ✅ pure (extra leakage = kept mask) |
| Ext 3 (composition selector) | `composition_selector.py` | ❌ — runs at deployment on plaintext | ❌ | ❌ | ❌ | ✅ pure |
| Ext 1 (bootstrap scheduler) | `bootstrap_scheduler.py` | ❌ — schedule is public | ❌ | ❌ | ❌ | ✅ pure |
| Ext 2 (learnable poly) | `poly/learnable.py` | ❌ — coefficients frozen at deploy | ❌ | ❌ | ❌ | ✅ pure |
| Ext 4 (RoBERTa, DistilBERT) | `models/backbone.py` | ❌ | ❌ | ❌ | ❌ | ✅ pure |

## Out-of-scope attacks

We do **not** address:
- malicious server (active deviation from protocol) — requires zk-SNARKs
  on top, well beyond this thesis
- side-channel attacks on the client's secret key
- model extraction via repeated queries (orthogonal — see Tramèr et al.)
- adversarial inputs (orthogonal — robustness)

These are explicitly listed as out-of-scope in the paper.

## Paper section sketch (for §6 Threat Model)

1. **System & assumptions** — single-server semi-honest, FHE-pure
2. **Adversary capabilities** — what the server sees + can compute
3. **Information leakage** — table of what's revealed (ciphertext count,
   circuit shape, kept-mask if Ext W content_teacher)
4. **What's protected** — formal statement: any two queries `x, x'` of
   the same length are computationally indistinguishable from the
   server's view
5. **Comparison** — table above vs BOLT/Iron/MPCFormer/NEXUS
6. **Out-of-scope** — list above
