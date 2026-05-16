# AFT — An Attention Free Transformer

**Citation**: Zhai, Talbott, Srivastava, Huang, Goh, Zhang, Susskind. *arXiv 2105.14103*, 2021.

**One-line summary**: Replaces dot-product attention with a query-conditioned but attention-free mixing rule that combines keys and values with learned positional biases in linear complexity.

## Core contribution

- Eliminates explicit dot-product self-attention.
- Combines key/value content with learned positional bias, then gates the result elementwise with the query.
- Targets long-context efficiency while preserving global connectivity.

## Key technical mechanism

- Linear memory in both context size and hidden dimension.
- Still contains **query-conditioned gating** at inference time.

## Threat model / setting

Plaintext efficient-Transformer paper.

## Relevance to Synthesizer-LPAN

- AFT is closer to our direction than Linformer/Performer because it removes explicit dot-product attention.
- However, it still keeps query-conditioned behavior in the runtime mixer, so it does not collapse cleanly to a fixed plaintext operator in FHE.
- That makes it less attractive than Synthesizer for the current encrypted cost model.

## Direct comparison points / how to cite

- "AFT [Zhai et al., 2021] removes dot-product attention but still gates the token mixer with the query at inference time; Synthesizer-LPAN goes further by freezing the full mixing pattern offline and removing query dependence from the encrypted attention path."

## Decision outcome

Worth mentioning as a near-neighbor, but not better than Synthesizer for the present thesis.