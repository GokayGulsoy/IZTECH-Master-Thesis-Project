# MPCFormer — Fast, Performant and Private Transformer Inference with MPC

**Citation**: Li, Shao, Wang, Guo, Xing, Zhang. *ICLR 2023*. CMU / MBZUAI / UC Berkeley. Code: github.com/MccRee177/MPCFormer.

**One-line summary**: Plug-in MPC-friendly approximations (e.g., quadratic softmax) + knowledge distillation to recover accuracy lost from approximation.

## Core contribution

Two-stage framework:
1. **Plug-in approximations** — replace bottleneck non-linears (softmax, GELU) with MPC-friendly polynomial / quadratic forms.
2. **Knowledge distillation** from the original fine-tuned model to recover accuracy (intermediate-layer matching, à la TinyBERT/PKD).

Designed atop CrypTen, but the recipe is engine-agnostic.

## Key numbers

- IMDb: **5.3×** faster than BERT-base at matched accuracy; **5.9×** faster than BERT-large at matched accuracy.
- GLUE: **97% of BERT-base accuracy** at **2.2×** speedup.
- Vanilla BERT-base in MPC takes ~60 s; un-encrypted takes <1 s — illustrates the gap MPCFormer aims to close.

## Threat model

2PC with optional trusted third party (CrypTen-style), interactive. Same security level as Iron / BOLT.

## Relevance to HyPER-LPAN

- **Validates the "approximate, then distill" recipe** that we use in Ext 1 (LinearMixing) and Ext 2 (LPAN).
- Confirms that **quadratic / low-degree approximations** of softmax are tractable when paired with KD — supports our QuadAttention primitive choice.
- They prove KD recovers accuracy from approximation; we use the same distillation training pipeline.

## Direct comparison

- "MPCFormer [Li et al., ICLR'23] showed that aggressive MPC-friendly approximations can be paired with knowledge distillation to recover accuracy. We adopt the same training paradigm but for FHE-friendly primitives, and additionally introduce a layer-wise composition selector that chooses approximation aggressiveness per layer rather than uniformly."
