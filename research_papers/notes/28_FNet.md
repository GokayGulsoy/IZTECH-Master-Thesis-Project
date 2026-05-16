# FNet — Mixing Tokens with Fourier Transforms

**Citation**: Lee-Thorp, Ainslie, Eckstein, Ontañón. *NAACL 2022*.

**One-line summary**: Replaces self-attention in encoder models with a fixed Fourier token mixer, preserving much of BERT's task accuracy while greatly improving plaintext throughput.

## Core contribution

- Shows that encoder-style NLP tasks can tolerate fixed token mixing surprisingly well.
- Replaces self-attention with an unparameterized Fourier transform.
- Achieves **92--97%** of BERT accuracy on GLUE while training about **80% faster** on GPUs and **70% faster** on TPUs at standard lengths.

## Key technical mechanism

- Token mixing is a fixed linear transform, not a data-dependent attention map.
- No query-key interactions, no softmax, and no runtime attention probabilities.

## Threat model / setting

Plaintext efficient-encoder paper.

## Relevance to Synthesizer-LPAN

- This is the strongest alternative future architecture branch for our project.
- Like Synthesizer, it replaces attention with a fixed linear mixer, making it naturally attractive for FHE.
- Unlike Synthesizer, it uses a universal, non-task-specific transform rather than a learned task-specific pattern. That could simplify deployment but may sacrifice more accuracy.

## Direct comparison points / how to cite

- "FNet [Lee-Thorp et al., NAACL'22] is the strongest alternative fixed-mixer baseline to compare against Synthesizer-LPAN: both remove query-key interactions, but Synthesizer retains a learned task-specific mixing matrix."
- "If we explore a second architecture family after the core thesis sweep, FNet is the cleanest candidate because it is also an input-independent linear operator under FHE."

## Decision outcome

Do not block the current training plan on FNet, but keep it as the first future pilot after the main Synthesizer-LPAN results are complete.