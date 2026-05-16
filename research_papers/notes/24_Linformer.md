# Linformer — Self-Attention with Linear Complexity

**Citation**: Wang, Li, Khabsa, Fang, Ma. *arXiv 2006.04768*, 2020.

**One-line summary**: Approximates self-attention as low-rank and projects the sequence dimension to reduce attention complexity from $O(n^2)$ to $O(n)$ in plaintext.

## Core contribution

- Argues the self-attention matrix is low rank.
- Applies learned projections on the sequence dimension of $K$ and $V$.
- Preserves near-Transformer accuracy while improving plaintext memory and runtime.

## Key technical mechanism

- Sequence-axis projection $L \to k$ turns $QK^\top$ from quadratic to linear in sequence length.
- Keeps query-conditioned attention; only compresses it.

## Threat model / setting

Plaintext efficient-attention paper.

## Relevance to Synthesizer-LPAN

- Important negative comparison point: it looks attractive on paper but is a poor fit for the current CKKS packing/layout.
- Under NEXUS-style column-major packing, sequence-axis projections introduce expensive rotations that erase much of the algebraic savings.
- We already observed this locally: only a modest projected win, not a compelling pivot.

## Direct comparison points / how to cite

- "Linformer [Wang et al., 2020] reduces quadratic attention through low-rank sequence projection, but still retains query-conditioned attention and requires token-axis mixing that is expensive in our CKKS packing regime."
- "We evaluated the Linformer direction conceptually and found it inferior to Synthesizer for our FHE cost model because projection along the sequence axis reintroduces costly rotations."

## Decision outcome

Do not pivot to Linformer before the current training sweep.