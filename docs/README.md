# Synthesizer-LPAN — Project Documentation

> **Synthesizer-LPAN** = **Synthesizer Learnable Polynomial Activation Network**
> Single-GPU sub-100 s end-to-end coherent FHE BERT inference via architectural attention elimination.

Living reference for the Synthesizer-LPAN FHE Transformer thesis project.
Read these in order when starting a new conversation or onboarding.

| Doc | Purpose |
|---|---|
| [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) | Thesis goals, headline result, contributions |
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | Synthesizer-LPAN design, math, algorithm |
| [02_FHE_PROTOCOL.md](02_FHE_PROTOCOL.md) | HEonGPU CKKS configuration, depth budget, packing |
| [03_THREAT_MODEL.md](03_THREAT_MODEL.md) | Pure non-interactive FHE, leakage analysis |
| [04_OPTIMIZATIONS.md](04_OPTIMIZATIONS.md) | BSGS-fused mask×diag, batching, chain tuning |
| [05_REPRODUCING_RESULTS.md](05_REPRODUCING_RESULTS.md) | Build, benchmark, correctness scripts |
| [06_HARDWARE.md](06_HARDWARE.md) | H100 single-GPU, vendored HEonGPU build |
| [07_REPO_LAYOUT.md](07_REPO_LAYOUT.md) | Module map, branch structure |
| [08_TRAINING_EXPERIMENT_ROADMAP.md](08_TRAINING_EXPERIMENT_ROADMAP.md) | Pre-pod training order, gates, and experiment matrix |
| [TECHNIQUES_JOURNEY.md](TECHNIQUES_JOURNEY.md) | What was tried, what failed, why we landed at Synthesizer-LPAN |

For the paper-by-paper literature notes and the current claim-safe related-work
matrix, see [research_papers/notes/00_INDEX.md](../research_papers/notes/00_INDEX.md).

## Headline result (May 2026)

| Configuration | Wall-time / 12-layer fwd | Speedup vs honest LPAN baseline |
|---|---|---|
| Honest LPAN (full softmax-poly, all 12 layers) | 833 s | 1.00× |
| **Synthesizer-LPAN + BSGS, BATCH=16, chain=22** | **60.9 s** | **13.67×** |

Single H100 SXM5, HEonGPU CKKS, ring N=2¹⁶, scale 2⁴⁰, sequence length L=128.

## Comparison with concurrent work

| System | Hardware | Wall-time | Threat model | Notes |
|---|---|---|---|---|
| **Synthesizer-LPAN (this work)** | **1× H100** | **60.9 s** | pure FHE | architectural breakthrough |
| CERIUM (Dec 2025, CMU+NVIDIA) | 8× B200 | 8.8 s | pure FHE | multi-GPU framework, ~$200K cluster |
| CERIUM | 1× H100 | 36.1 s | pure FHE | (same architecture as plain BERT) |
| CERIUM | 1× A100 | 66 s | pure FHE | |
| NEXUS (Crypto'24) | 1× GPU | ~7 s/sample reported (different setup) | pure FHE | |

CERIUM is a **framework**-level optimization (DSL + compiler + runtime).
Synthesizer-LPAN is an **architectural** contribution that eliminates Wq, Wk,
Q·Kᵀ, and softmax-poly entirely. The two are orthogonal — CERIUM's runtime
could compile our circuit; our architecture could replace any plain-BERT
front-end inside CERIUM. They are complements, not competitors.

## Branch state (May 2026)

```
* synthesizer-lpan-production    ← HEAD (this docs reflects)
  feature/hyper-lpan-extensions  ← prior, archived
  feature/ckks-protocol          ← validated baseline
  main
```

Production branch invariants: vendored `third_party/HEonGPU/` (8 MB,
commit pinned in `third_party/HEonGPU.commit`), modular
`fhe_thesis/encryption/{attention,linear,layernorm,colmajor,multi}.py`,
no NEXUS-suffixed public APIs.
