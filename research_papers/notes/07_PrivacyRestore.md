# PrivacyRestore — Privacy-Preserving LLM Inference via Privacy Removal & Restoration

**Citation**: Zeng, Wang, Yang, Lu, Li, Zhuang, Chen. *arXiv 2406.01394v5* (May 2025). South China University of Technology.

**One-line summary**: Client removes privacy spans + applies dχ-privacy noise + sends a "meta restoration vector"; server pre-trains restoration vectors per privacy span type and reconstructs.

## Core contribution

Plug-and-play protocol for client-server LLM inference:
1. **Offline (server)**: train restoration vectors for each privacy span type.
2. **Online (client)**: identify privacy spans → strip from query → aggregate restoration vectors into a meta vector → apply dχ-privacy → send remaining query + noisy meta vector to server.
3. **Inference (server)**: run LLM with restoration vector injected.

Provably bounds the privacy budget against linear growth typical of word-level DP.

## Key numbers

Evaluated on medical and legal domains. Maintains acceptable utility while preventing the linear privacy-budget growth of word-level dχ-privacy methods (Feyisetan et al., Mattern et al.).

## Threat model

**Differential privacy + plaintext server computation.** Server sees the (sanitized) query in cleartext. **Much weaker** than FHE — relies on DP noise to obscure private spans rather than cryptographic confidentiality of the entire input.

## Relevance to HyPER-LPAN

- Represents the **DP end of the privacy spectrum**; we sit at the **FHE end**.
- Strengthens our claim of "strongest threat model among practical proposals" by giving a concrete weaker-but-faster baseline.
- Demonstrates that the field is actively exploring trade-offs across the DP–MPC–TEE–FHE spectrum.

## Direct comparison

- "Recent work such as PrivacyRestore [Zeng et al., 2025] explores DP-based privacy for LLM inference: the client perturbs sensitive spans with dχ-privacy and the server runs the model in plaintext. This trades cryptographic guarantees for inference speed. HyPER-LPAN takes the opposite design point: cryptographic confidentiality of the entire input under FHE, with non-interactive server computation."

## Why we don't compete head-to-head

Different threat models. Apples to oranges. We position both in our threat-model table, not in our latency comparison.
