# PoWER-BERT — Accelerating BERT Inference via Progressive Word-vector Elimination

**Citation**: Goyal, Choudhury, Raje, Chakaravarthy, Sabharwal, Verma. *ICML 2020*. IBM Research. Code: github.com/IBM/PoWER-BERT.

**One-line summary**: Eliminate redundant word-vectors **progressively** through BERT layers using attention-based significance scores — **4.5× speedup** with **<1% accuracy loss** on GLUE.

## Core contribution

- **Diffusion of information** observation: as word-vectors pass through transformer blocks, they progressively carry similar information (cosine similarity rises with depth). The CLS-token's sole role is also questioned — other positions work nearly as well (mean drop 1.2% on SST-2).
- Three-step training: (1) standard fine-tune, (2) length-configuration search via differentiable retention parameters, (3) re-train with frozen length config.
- Concrete schedule for BERT-base on SST-2: 12 layers keep `[80, 73, 70, 50, 50, 40, 33, 27, 20, 15, 13, 3]` vectors out of 128.
- **6.8× speedup** when applied on top of ALBERT (compressed BERT).

## Why it works

Self-attention diffuses information across positions; redundant vectors carry the same content as their neighbors. Significance score = total attention received from other tokens.

## Limitations

- **Sequence-level classification only** — eliminates word-vectors so token-level tasks (NER, QA) impossible without modification (fixed by Length-Adaptive).
- Requires **separate model per latency budget** (also fixed by Length-Adaptive).
- **Not directly FHE-compatible**: dynamic per-input shape, attention-score-based selection requires comparison + sort under encryption.

## Relevance to HyPER-LPAN — Ext W (PoWER-LPAN)

- **Direct prior art** for our Ext W proposal.
- Same insight (information diffusion → token redundancy) but applied to **FHE cost reduction** rather than plaintext FLOPs.
- Differences:
  - We must use **input-independent** (or significance-precomputed) keep masks because dynamic shapes break FHE.
  - We define keep-mask per layer offline (per task or per prompt template), then pad to constant length and zero-out dropped positions.
  - We can compose with our composition selector: dropped positions cost zero LPAN/Quad/LM operations.

## Direct citation use

- "Our Ext W (PoWER-LPAN) extends PoWER-BERT [Goyal et al., ICML'20] to the FHE setting. PoWER-BERT eliminates word-vectors based on dynamic significance scores; we instead pre-compute a static keep-schedule offline (by attention-score profiling on a calibration set) so that the per-layer cipher-tensor shape stays static — a hard requirement under FHE."

## Numbers we need

- Layer-wise keep schedule (per task) is the headline output of our Ext W.
- Target: match PoWER-BERT's 4.5× FLOP reduction in **token count**, then translate to FHE-cost reduction via our composition selector.
