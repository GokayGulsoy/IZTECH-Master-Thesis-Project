# Thesis Background — FHE Polynomial Approximations Blueprint

**Type**: Internal blueprint document (not a research paper). Author: own / advisor planning notes.

**One-line summary**: 4-step literature-review outline for §2.3 (Non-Linearity Bottleneck) and §3.3 (Polynomial Approximations) of the master's thesis.

## The 4 steps

1. **Polynomial baseline**: CryptoNets — `f(x) = x²` substitution and accuracy/efficiency trade-off.
2. **Softmax & attention bottleneck**: Iron + THE-X — softmax-denominator approximation, Taylor failure outside small radius, perplexity/BLEU impact.
3. **Minimax & Chebyshev** (theoretical core): Chebyshev avoids Runge's phenomenon; Remez algorithm produces minimax-optimal polynomial.
4. **Compiler architecture**: CHET (Dathathri et al., 2019) — automated tensor routing, weight quantization, FHE circuit generation.

## What this document gives us

- **Section structure** for thesis Chapters 2 and 3.
- **Required citations**: CryptoNets, Iron, THE-X, CHET, plus Zama / Chillotti minimax papers.
- A **meta-prompt** to use with Copilot for drafting LaTeX from these papers.

## Action items derived from this doc

1. Draft thesis §2.3 covering: ReLU → x² → polynomial GELU/softmax history.
2. Draft thesis §3.3 covering: Taylor vs. Chebyshev vs. Remez minimax, with formulas.
3. Decide whether to cite CHET / a modern FHE compiler — likely **yes**, since we use OpenFHE which has compiler-style optimizations internally.

## Relevance

This is a **planning artefact**, not a citation. Its role is to keep our literature-review structure consistent with what was originally scoped.
