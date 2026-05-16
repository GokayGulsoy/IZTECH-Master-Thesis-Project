# Nyströmformer — A Nyström-Based Algorithm for Approximating Self-Attention

**Citation**: Xiong, Zeng, Chakraborty, Tan, Fung, Li, Singh. *AAAI 2021*.

**One-line summary**: Uses landmark-based Nyström approximation to reduce self-attention complexity to linear time while preserving competitive plaintext accuracy.

## Core contribution

- Approximates the softmax attention matrix with a low-rank landmark decomposition.
- Scales to longer sequences while remaining competitive on GLUE and long-range tasks.
- Keeps the attention mechanism itself rather than replacing it with a fixed mixer.

## Key technical mechanism

- Selects landmark summaries and reconstructs attention with a Nyström approximation.
- Still relies on query-conditioned attention and approximate matrix inversion / reconstruction structure.

## Threat model / setting

Plaintext efficient-attention paper.

## Relevance to Synthesizer-LPAN

- Important comparison point because it is another strong linear-time attention baseline.
- Like Linformer and Performer, it retains data-dependent attention and therefore does not align with the core FHE objective of eliminating the encrypted attention path.
- The landmark-reconstruction structure is not obviously friendlier than frozen $A_h V_h$ under the current HE kernels.

## Direct comparison points / how to cite

- "Nyströmformer [Xiong et al., AAAI'21] approximates softmax attention rather than removing it; our work instead freezes the mixing pattern offline so the encrypted circuit contains only a plaintext-times-ciphertext linear map."

## Decision outcome

Do not pivot before the current task sweep.