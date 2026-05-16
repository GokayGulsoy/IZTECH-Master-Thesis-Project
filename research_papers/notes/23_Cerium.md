# Cerium — A Scalable Multi-GPU Framework for Encrypted Large-Model Inference

**Citation**: Jayashankar, Kim, Sullivan, Zheng, Skarlatos. *arXiv 2512.11269*, December 2025.

**One-line summary**: Multi-GPU compiler/runtime stack for pure-FHE inference that outperforms hand-written GPU code and currently defines the latency frontier for encrypted large-model inference.

## Core contribution

- Integrates a DSL, optimizing compiler, and runtime for FHE inference on NVIDIA GPUs.
- Adds communication-aware parallelization, sparse polynomial handling, new data layouts, and automatic kernel generation.
- Extends beyond BERT to large-model settings such as Llama3-8B.

## Key technical mechanism (numbers)

- Reports **36.1 s** for BERT-Base at $L=128$ on **1x H100**.
- Reports **8.8 s** for BERT-Base at $L=128$ on **8x B200**.
- Reports **66.0 s** on **1x A100**.
- Reports up to **2.25x** speedup over expert-written GPU libraries.
- Reports **7.5 ms** bootstrapping, the first GPU result under 10 ms.

## Threat model / setting

Pure FHE, non-interactive, same broad threat-model class as ours.

## Relevance to Synthesizer-LPAN

- Cerium **supersedes any latency-SOTA claim** we might otherwise make.
- Cerium is a **system/compiler** contribution around standard attention; Synthesizer-LPAN is an **architectural** contribution that changes the attention block itself.
- The two are complementary: Cerium could in principle compile a Synthesizer-LPAN front end.

## Direct comparison points / how to cite

- "Cerium [Jayashankar et al., 2025] is the current pure-FHE latency leader, reporting 36.1 s on a single H100 and 8.8 s on 8x B200 for BERT-Base at 128 tokens."
- "Our claim is therefore not current latency SOTA, but an orthogonal architectural redesign that removes the expensive encrypted attention path itself."
- "Synthesizer-LPAN should be framed as complementary to Cerium: architecture-level cost reduction rather than compiler/runtime optimization."

## What we should not claim

- Do **not** claim current single-GPU latency SOTA.
- Do **not** claim current global sub-100 pure-FHE novelty without qualification.

## Decision outcome

Use Cerium as the top line in every current related-work table and pivot our framing to architectural novelty plus honest baseline speedup.