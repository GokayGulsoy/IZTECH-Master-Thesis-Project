# THE-X — Privacy-Preserving Transformer Inference with HE

**Citation**: Chen, Bao, Huang, Dong, Jiao, Jiang, Zhou, Li, Wei. *Findings of ACL 2022*. Beihang + Microsoft Research.

**One-line summary**: HE-only transformer inference that **replaces non-linears with cheap proxies** (ReLU instead of GELU, ReLU+poly instead of softmax) — accepts intermediate-layer leakage to clients to keep things fast.

## Core contribution

Workflow that converts a fine-tuned transformer into an FHE-evaluable function by:
- GELU → ReLU
- Softmax → ReLU + polynomial
- LayerNorm → simplified polynomial form
- Negligible accuracy drop reported on downstream NLP tasks.

## Threat model — important caveat

**Inputs to each non-linear layer are revealed to the client**, who computes the non-linear in plaintext and returns the result. Iron explicitly criticizes this: "the inputs of each non-linear layer are leaked to the client, which may cause severe privacy leakages in real-world applications."

**Strictly weaker** than ours, NEXUS, Iron, and BOLT.

## Relevance to HyPER-LPAN

- Cite as the **first attempt** at HE-only transformer inference, but with a critical security flaw.
- HyPER-LPAN evaluates non-linears **inside FHE** (no plaintext leakage), matching the rigor of NEXUS.
- Justifies the need for proper polynomial approximations (LPAN, QuadAttention) rather than client-side computation hacks.

## Direct comparison

- "THE-X [Chen et al., ACL'22] demonstrated that crypto-friendly substitutions can preserve accuracy, but it leaks all non-linear-layer inputs to the client. HyPER-LPAN evaluates **all** non-linears under FHE, matching the rigorous threat model of NEXUS while incurring lower per-token cost via the layer-wise composition design."
