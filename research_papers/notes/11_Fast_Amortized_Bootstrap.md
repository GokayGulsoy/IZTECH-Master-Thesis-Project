# Fast Amortized Bootstrapping with Small Keys and Polynomial Noise Overhead

**Citation**: Guimarães, Pereira. *EUROCRYPT 2024* (and follow-ups). IMDEA Software Institute + UNICAMP.

**One-line summary**: Amortized bootstrapping for FHE based on lattice problems with **polynomial** approximation factor (vs. superpolynomial in CKKS/BGV) — **2 to 8-bit messages bootstrapped in 1.46–28.5 ms**.

## Core contribution

- New amortized bootstrapping where per-message homomorphic operations are `O(h)` and noise overhead is `O(√h · λ · log λ)` (h = Hamming weight of LWE key, λ = security parameter).
- Based on efficient homomorphic evaluation of **sparse polynomial multiplication**.
- Bootstrapping keys **47.5× smaller** than TFHE-rs.
- **2.5 to 38.7× faster** than TFHE-rs.

## Why it matters

- TFHE/FHEW family historically had: cheap bootstrap **per gate** but no SIMD, weak security assumption (polynomial γ).
- CKKS/BGV: SIMD bootstrap + strong security but **superpolynomial** lattice approximation factor and slow bootstrap.
- This paper bridges: **SIMD-style amortization** with **polynomial-γ security** — strictly stronger security assumption.

## Relevance to HyPER-LPAN

- **Future work direction**: if amortized bootstrapping becomes mature for real-valued (CKKS-style) data, our composition selector can be re-optimized with much cheaper bootstrap costs, allowing more LPAN layers.
- Also relevant if we ever target **integer-domain quantized transformers** (TFHE-style); our LinearMixing primitive could be evaluated bit-decomposed with this scheme.
- Cite in §7 (future work / discussion of bootstrap evolution).

## Direct citation use

- "Recent advances in amortized bootstrapping [Guimarães & Pereira, EUROCRYPT'24] suggest the per-bootstrap cost may drop substantially in the next generation of FHE libraries; HyPER-LPAN's composition-selector will translate any such cost reduction into either lower latency or higher accuracy at the same budget."
