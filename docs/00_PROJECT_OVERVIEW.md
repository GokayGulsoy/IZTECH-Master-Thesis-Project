# 00 — Project Overview

## Thesis title (working)

> **HyPER-LPAN: A Task-Adaptive, Pure-FHE Transformer Architecture for
> Privacy-Preserving NLP Inference**

İzmir Institute of Technology (İYTE) — Master's thesis,
candidate: Gökay (current semester: thesis-1, May 2026).

## Goals

1. **Pure non-interactive FHE** transformer inference under CKKS — no MPC,
   no mid-circuit decryption, no TEE. Stronger threat model than BOLT/Iron
   (2PC) and Iron-LM (MPC).
2. **GLUE accuracy within 1–3 points of plaintext** on SST-2 / MRPC / QNLI
   / RTE for BERT-base, RoBERTa-base, DistilBERT.
3. **Sub-10 s end-to-end latency** per sample on a 32-vCPU CPU pod with
   AVX-512 + HEXL acceleration. Stretch target: 5–7 s.
4. **Task-adaptive composition** — let the network decide per-layer
   whether to use LinearMixing / QuadAttention / LPAN (the headline
   contribution).

## Target venues

| Venue | Deadline | Track | Notes |
|---|---|---|---|
| **USENIX Security 2027** | Feb 2027 | full paper | primary; matches threat-model framing |
| **EMNLP 2027** | Jun 2027 | long paper | secondary; ML/NLP framing |
| ICLR 2027 (workshop) | TBD | poster | fallback for early visibility |

## Timeline

~7 months total: this semester (May–Jul 2026) → summer (Aug–Sep) →
next semester (Oct–Jan 2027) → submit Feb 2027.

Spent so far: **architecture validated** + **5 extensions + cleanup**
landed (commits `d288662 → e3845dd` on `feature/hyper-lpan-extensions`).

Remaining big rocks:
1. Full GLUE training (4 tasks × 3 backbones × 3 seeds)
2. Pareto plot harness (FHE latency vs accuracy)
3. Pod latency benchmark (validate 5–7 s target)
4. Threat-model formalization for paper
5. Thesis chapters 3 (methodology) + 4 (results) write-up

## Key validated numbers (carry forward)

| Metric | Value | Source |
|---|---|---|
| SST-2 HyPER-LPAN | 90.83 % | `feature/ckks-protocol` @ `14a43c2` |
| MRPC v1 (LM[0-3]+Q[4-7]) | 82.60 F1 | uncompetitive — selector v2 needed |
| Per-layer ct×ct ops | 126 → 84 (-33 %) | Phase 2a |
| Per-layer pt×ct ops | 96 → 108 (+12 %) | acceptable trade |
| CKKS depth (canonical) | 348 (was 396, -12 %) | Phase 2a |

## Locked design decisions

- **Order of operations**: re-modularize → FHE opts → benchmarks
- **Tasks**: SST-2, MRPC, QNLI, RTE (matches MPCFormer/BOLT/NEXUS)
- **Models**: BERT-base, RoBERTa-base, DistilBERT
- **Ablations**: layer composition, stage ordering, poly degrees,
  co-adaptation, KD γ sweep, num heads in QuadAttention
- **Hardware**: training local on MSI RTX 5070 Ti or RunPod H100
  (decision per task — see [06_HARDWARE.md](06_HARDWARE.md));
  FHE inference on 32-vCPU Threadripper 7960X Pod.

## Critical constraint

**Everything must remain pure non-interactive FHE.** No design choice
that requires the server to decrypt mid-circuit, talk to the client
mid-inference, or run inside a TEE is acceptable. See
[03_THREAT_MODEL.md](03_THREAT_MODEL.md) for the audit table.
