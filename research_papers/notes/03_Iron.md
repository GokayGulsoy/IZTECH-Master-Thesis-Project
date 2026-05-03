# Iron — Private Inference on Transformers

**Citation**: Hao, Li, Chen, Xing, Xu, Zhang. *NeurIPS 2022*. UESTC + NTU.

**One-line summary**: Hybrid HE + 2PC framework providing rigorous protocols for matmul, Softmax, GELU, LayerNorm — first end-to-end full-Transformer private inference baseline.

## Core contribution

- **Compact packing** for HE matrix multiplication: `√m×` less communication than Cheetah's matrix-vector extension (m = output rows).
- **SIRNN-based** protocols for Softmax / GELU / LayerNorm, optimized to ~half the runtime/comm of prior art.
- **Numerically precise** (no significant accuracy degradation vs plaintext).
- Formal security proofs.

## Key numbers

- **3–14×** less communication than SIRNN, **3–11×** less runtime.
- Up to **two orders of magnitude** improvement over MP-SPDZ.
- Implementation: BERT-Tiny / Medium / Base / Large on GLUE.
- BOLT measures Iron at **280.99 GB / 216 minutes** for one BERT-base on a 100 Mbps / 80 ms WAN.

## Threat model

Honest-but-curious, 2PC, **interactive**. Same as BOLT, weaker than ours.

## Relevance to HyPER-LPAN

- Often cited as "the foundational hybrid baseline" in MPC-for-transformers. We cite as historical context and as the system NEXUS+BOLT both surpass.
- Iron's matrix-multiplication packing is conceptually superseded by NEXUS's compression/decompression for the non-interactive setting.

## Direct comparison

- "Iron [Hao et al., NeurIPS'22] established the first end-to-end private BERT inference with rigorous protocols for all Transformer non-linears, but its 280 GB / 216 min cost per inference makes it impractical and motivated subsequent communication-focused work (BOLT, Bumblebee, NEXUS)."

## Supplemental file

`Iron Private Inference on Transformers - Supplemental.txt` contains formal security games, additional experiments, and protocol pseudocode. No additional architectural insight beyond the main paper.
