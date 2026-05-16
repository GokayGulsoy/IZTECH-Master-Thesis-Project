# Synthesizer — Rethinking Self-Attention in Transformer Models

**Citation**: Tay, Bahri, Metzler, Juan, Zhao, Zheng. *ICML 2021*.

**One-line summary**: Shows that competitive Transformer performance does not require token-token query-key interactions; synthetic, learned attention patterns can replace dot-product attention surprisingly well.

## Core contribution

- Reframes self-attention as a learned mixing pattern rather than a mandatory query-key interaction.
- Introduces several Synthesizer variants, including random, dense, and factorized synthetic attention.
- Demonstrates that synthetic attention is competitive with vanilla Transformers across GLUE, SuperGLUE, language modeling, and generation tasks.

## Key technical mechanism (numbers)

- Replaces data-dependent attention weights with learned synthetic mixing weights.
- Reports that simple Random Synthesizer is about **60% faster** than Dynamic Convolutions while improving perplexity by a relative **3.5%**.
- Reports that factorized Synthesizer variants can outperform Linformer on encoder-only tasks.

## Threat model / setting

Plaintext Transformer architecture paper. No privacy or cryptography component.

## Relevance to Synthesizer-LPAN

- This is the **architectural parent** of the thesis contribution.
- Among efficient-attention families, Synthesizer is the cleanest fit for FHE because it can be frozen into a plaintext matrix $A_h$ and therefore removes $W_q$, $W_k$, $QK^\top$, and runtime Softmax from the encrypted circuit.
- It justifies the central claim that query-key interactions are not sacred when the downstream cost model changes radically.

## Direct comparison points / how to cite

- "Synthesizer [Tay et al., ICML'21] showed that competitive Transformer performance can be achieved without token-token query-key interactions; we port this architectural idea to FHE, where its cost advantage becomes far larger than in plaintext."
- "Plaintext Synthesizer was proposed for efficiency, but under FHE it becomes an architectural elimination of the most expensive encrypted primitive: data-dependent attention."
- "Our contribution is not a new efficient-attention family in plaintext; it is the first FHE realization of Synthesizer-style frozen mixing."

## Decision outcome

Keep Synthesizer-LPAN as the mainline. This is the strongest architectural justification for the thesis.