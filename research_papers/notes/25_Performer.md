# Performer — Rethinking Attention with Performers

**Citation**: Choromanski, Likhosherstov, Dohan, Song, Gane, Sarlos, Hawkins, Davis, Mohiuddin, Kaiser, et al. *ICLR 2021*.

**One-line summary**: Approximates softmax attention with positive orthogonal random features (FAVOR+) to achieve linear-time, linear-space query-conditioned attention with theoretical guarantees.

## Core contribution

- Rewrites softmax attention as a kernel approximation.
- Uses FAVOR+ random features to obtain unbiased or nearly unbiased estimators of attention.
- Keeps the semantic behavior of query-conditioned attention while reducing plaintext asymptotic cost.

## Key technical mechanism

- Softmax kernel approximation via positive orthogonal random features.
- Linear-time attention without assuming sparsity or low rank.
- Still requires query-dependent feature maps and normalization terms.

## Threat model / setting

Plaintext efficient-attention paper.

## Relevance to Synthesizer-LPAN

- Performer is strong plaintext prior art, but not a better FHE pivot for the current system.
- It still computes query-dependent attention features at inference time, so it does **not** remove the encrypted dependence on the input the way Synthesizer does.
- For CKKS, the extra feature-map products and reductions are less attractive than a frozen plaintext mixing matrix.

## Direct comparison points / how to cite

- "Performer [Choromanski et al., ICLR'21] keeps query-conditioned attention and linearizes it through random-feature kernels; Synthesizer-LPAN instead removes query-conditioned attention from the encrypted circuit entirely."
- "Efficient-attention methods designed for plaintext asymptotics do not automatically improve the FHE cost model, where fixed plaintext linear maps are substantially cheaper than query-dependent kernels."

## Decision outcome

Useful related work, but not a pre-pod pivot candidate.