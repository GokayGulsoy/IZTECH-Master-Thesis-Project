# PEGASUS — Bridging Polynomial and Non-polynomial Evaluations in HE

**Citation**: Lu, Huang, Hong, Ma, Qu. *IEEE S&P 2021*. Alibaba Gemini Lab + UPenn.

**One-line summary**: Practical framework that switches **between CKKS (word-wise SIMD) and FHEW (bit-wise LUT) ciphertexts without decryption**, enabling efficient evaluation of both arithmetic and non-polynomial functions.

## Core contribution

- **FHEW → CKKS conversion** with sublinear (rather than linear) computational complexity.
- Conversion-key size reduced from **80 GB → 12 MB**.
- Demonstrates secure: sigmoid, ReLU, min/max, division, sorting, max-pooling, decision-tree evaluation, K-means clustering (14–20× faster than prior best).

## Trade-off framework

Word-wise HE (CKKS):
- ✅ SIMD (multiple plaintexts per ciphertext).
- ❌ Hard for non-polynomial functions (sigmoid, min/max, division).

Bit-wise HE (FHEW/TFHE):
- ✅ Arbitrary functions via LUTs.
- ❌ No SIMD; slow on arithmetic; ~half-minute to multiply two 16-bit integers.

PEGASUS lets you have both.

## Relevance to HyPER-LPAN

- **Alternative architectural direction** that HyPER-LPAN explicitly does **not** take: we keep everything in CKKS and approximate non-linears with polynomials.
- If polynomial approximation ever becomes the bottleneck, switching to FHEW for non-linears (PEGASUS-style) is a natural next step. Cite in §7 (future work).
- Reinforces the design choice: polynomial-only approach (ours) avoids the conversion overhead but caps the achievable approximation quality.

## Direct citation use

- "PEGASUS [Lu et al., S&P'21] sidesteps the polynomial-approximation cost by converting between CKKS and FHEW ciphertexts. HyPER-LPAN takes the simpler all-CKKS path and absorbs the non-linearity cost via the layer-wise composition selector."
