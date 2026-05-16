# 03 — Threat Model

## Setting

Single-server semi-honest FHE inference, **single-round**, **non-interactive**:

```
client                                                     server
─────                                                     ──────
1. holds secret key sk
2. encrypts query x → enc(x)
3. sends enc(x) (and public sequence length L)            →
                                                            4. runs 12-layer
                                                               Synthesizer-LPAN
                                                               forward under
                                                               CKKS (HEonGPU)
                                                            5. returns
   ←  enc(logits)                                              enc(logits)
6. decrypts logits, takes argmax
```

The server **never** sees plaintext tokens, **never** decrypts
intermediate ciphertexts, **never** sends ciphertexts back to the
client mid-inference.

## Adversary capabilities (semi-honest)

The server:
- follows the protocol honestly
- but tries to learn as much as possible from what it sees
- has unbounded compute and storage
- knows the **public** model: architecture, weights $W_v, W_o, W_{\text{ffn}}$,
  the frozen attention pattern $A$, polynomial coefficients, chain length
- knows CKKS public parameters ($N$, scale, prime chain)

## Acceptable leakage

| Leakage | Why acceptable | Comparable to |
|---|---|---|
| Sequence length $L$ | required by every FHE-transformer protocol | NEXUS, BOLT, Iron, MPCFormer, CERIUM |
| Circuit shape (op counts, depth, chain) | model is public | universal |
| Wall-clock timing of ops | data-independent (no branch on ciphertext) | universal |

## What we do NOT leak

- Plaintext token values (`input_ids`)
- Embedding values
- Intermediate activations
- Attention weights *per query* (the frozen $A$ is public model state,
  but the per-query realisation $A \cdot V$ never leaves the ciphertext)
- Final logit values (only encrypted logits leave the server)
- Predicted class label (client decides)

## Comparison vs related work

| System | Crypto | Server holds sk? | Decrypts mid-circuit? | MPC rounds | Threat model |
|---|---|---|---|---|---|
| **Synthesizer-LPAN (this work)** | CKKS | ❌ | ❌ | 0 | strongest |
| CERIUM (Dec 2025) | CKKS | ❌ | ❌ | 0 | same as us |
| NEXUS (Crypto'24) | CKKS | ❌ | ❌ | 0 | same as us |
| BOLT (S&P'24) | CKKS + 2PC | ❌ | yes (via SS) | many | weaker |
| Iron (NeurIPS'22) | CKKS + 2PC | ❌ | yes | many | weaker |
| Iron-LM (USENIX'24) | CKKS + TEE | ❌ | yes (in TEE) | 0 | TEE-trusted |
| MPCFormer (ICLR'23) | secret sharing | distributed | yes | many | weaker |
| THE-X (ACL'22) | CKKS + client compute | ❌ | sends partial back | 1 | weaker |

**Synthesizer-LPAN matches NEXUS / CERIUM and dominates BOLT / Iron /
Iron-LM / MPCFormer / THE-X** on threat-model strength.

## Synthesizer-specific audit

The frozen attention pattern $A$ is the only architectural difference
from a standard FHE BERT. Security implications:

| Question | Answer |
|---|---|
| Does $A$ leak training data? | $A$ is the *average* attention pattern over a training set. Standard model-stealing leakage applies — same as any released model. |
| Can the server recover $V$ from $O = A V$? | $A$ is generally invertible, but $O$ is encrypted — server never sees $O$ in clear. |
| Does fixing $A$ across queries leak query-correlation? | $A$ is constant — every query yields a different encrypted $O$ that decrypts to a different value. No correlation visible to server. |
| Is the per-query attention realised at all? | Yes, mathematically: $O = A V$. But $O$ stays in CKKS ciphertext. |

## Out-of-scope attacks

Not addressed:

- malicious server (active deviation from protocol) — would require
  zk-SNARKs on top
- side-channel attacks on the client's secret key
- model extraction via repeated queries (orthogonal — see Tramèr et al.)
- adversarial inputs (orthogonal — robustness)
- multi-tenant timing channels on the server

These are explicitly listed as out-of-scope in the paper.

## Paper section sketch (for §6 Threat Model)

1. **System & assumptions** — single-server semi-honest, FHE-pure
2. **Adversary capabilities** — what the server sees + can compute
3. **Information leakage** — sequence length only
4. **What's protected** — formal statement: any two queries $x, x'$ of
   the same length are computationally indistinguishable from the
   server's view
5. **Synthesizer-specific audit** — the frozen $A$ adds no leakage
6. **Comparison table** — vs BOLT/Iron/MPCFormer/NEXUS/CERIUM
7. **Out-of-scope** — list above
