# BOLT — Privacy-Preserving Inference for Transformers

**Citation**: Pang, Zhu, Möllering, Zheng, Schneider. *IEEE S&P 2024*. CMU / UC Berkeley / TU Darmstadt. Code: github.com/Clive2312/BOLT.

**One-line summary**: SOTA **interactive** MPC transformer inference combining HE for matmul + 2PC for non-linears, with ML-side optimizations including oblivious word elimination.

## Core contribution

- **HE-only matmul** (no MPC for the heaviest linear ops) using baby-step giant-step rotations and an alternative ciphertext-plaintext matrix-matrix interpretation.
- **Polynomial pre-processing** for GELU(order 4) and Tanh(order 5): reduces multiplications from `n` to ~`⌈n/2⌉` (Horner-style improvement).
- **Oblivious word elimination** using attention-score-based ranking + bitonic sort — directly parallels PoWER-BERT but in the secure-computation setting.
- Secure-computation-aware fine-tuning to bridge fixed-point vs floating-point gap.

## Key numbers

- **10.91× less communication** than Iron (NeurIPS'22).
- **4.8 – 9.5× faster** than Iron across LAN/WAN.
- **59.61 GB** bandwidth and **10 509 interaction rounds** for one BERT-base inference (per NEXUS measurement).
- **$5.44/token** financial cost on AWS (per NEXUS analysis).

## Threat model

2PC, semi-honest, **interactive** (many rounds). **Strictly weaker** than ours and NEXUS — needs persistent client connection, leaks size of intermediate communication, infeasible on async or batch workloads.

## Relevance to HyPER-LPAN

- Strongest **MPC** baseline; we cite as "if you tolerate interaction, BOLT is the SOTA".
- Their oblivious word elimination is the analog of PoWER-BERT in MPC and validates that **token pruning is well-known**, justifying our Ext W positioning.
- Their polynomial pre-processing trick for GELU is independent of our work but compatible — could combine with our LPAN design.

## Direct comparison

- "Interactive MPC systems like BOLT [Pang et al., S&P'24] achieve 4.8–9.5× speedup over Iron but require 10 509 communication rounds and 59.6 GB bandwidth per inference, ruling out batch / async deployments. HyPER-LPAN targets the strictly stronger non-interactive setting."

## What we should NOT claim

- HyPER-LPAN is **not** faster wall-clock than BOLT on a fast LAN; the comparison axis is bandwidth/round-count and threat model, not raw seconds.
