# NEXUS — Secure Transformer Inference Made Non-Interactive

**Citation**: Zhang, Yang, He, Chen, Lu, Wang, Hou, Liu, Ren, Yang. *NDSS 2025*. Zhejiang University. Code: github.com/zju-abclab/NEXUS.

**One-line summary**: First **non-interactive** secure transformer inference protocol — pure RNS-CKKS, single round of communication, BERT inference in 37.3 s @ 164 MB on GPU.

## Core contribution

Builds an end-to-end protocol where the client sends **one** encrypted input and receives **one** encrypted prediction. No streaming back-and-forth. Same threat model as ours (semi-honest, FHE-only, no MPC mid-circuit decryption).

## Key technical mechanism (numbers)

- **SIMD ciphertext compression / decompression**: avoids wasted slots from sparse packing in matrix-matrix multiplication.
- **Amortization-friendly offline-online matrix multiplication** (Sec III.C).
- **SIMD slot folding** for non-linears.
- **Secure Argmax in O(log m)** sign-operations and rotations vs. Phoenix (CCS'22) `O(m)`. For BERT vocabulary `m = 30,522` and Llama-3 `m = 128,256` this is the difference between "feasible" and "infeasible".
- BERT-base, 128 tokens, 100 Mbps WAN, 80 ms latency:
  - **CPU**: 1.79× faster than Bumblebee, 98.1% communication saved, 2.38× cheaper.
  - **GPU**: 42.3× speedup, $0.05/token (vs BOLT's $5.44 — **109× cheaper**).
  - **Bandwidth**: 372.5× less than BOLT, 53.6× less than Bumblebee.
- Open-source implementation on both CPU and GPU.

## Threat model

Two-party, semi-honest, computationally bounded. **C learns nothing about S's model except prediction. S learns nothing about C's input.** Identical to ours. RNS-CKKS only — **no MPC, no TEE**. Same strongest-end-of-spectrum positioning as HyPER-LPAN.

## Relevance to HyPER-LPAN

- **Closest competitor**. Same threat model, same crypto primitive (RNS-CKKS), same target (BERT-class transformers).
- NEXUS attacks **systems-level efficiency** (better packing, better Argmax). HyPER-LPAN attacks **architectural cost reduction** (replace expensive softmax with cheaper attention primitives). **Orthogonal and composable.**
- Their secure Argmax (`O(log m)`) should be cited as the right tool for our final classification head — we should not reinvent it.

## Direct comparison points (ready-to-cite sentences)

- "NEXUS [Zhang et al., NDSS'25] is the closest prior work in our threat model and reports 37.3 s end-to-end BERT inference on GPU with 164 MB bandwidth. HyPER-LPAN is **orthogonal**: NEXUS optimizes the protocol around the standard transformer; we modify the architecture to lower the per-layer FHE cost while preserving accuracy."
- "We adopt the secure Argmax of NEXUS for our final classification head, contributing `O(log m)` sign operations rather than `O(m)`."
- "Unlike BOLT (interactive, 10 509 rounds, 59.61 GB) and Bumblebee, both NEXUS and HyPER-LPAN are strictly non-interactive."

## Open questions to flag

- NEXUS reports CPU/GPU; we report CPU only (Threadripper Pod) — we must avoid head-to-head GPU claims unless we run on equal hardware.
- Their secure Argmax assumes logit decryption returns only Argmax (membership-inference defense). We should adopt the same convention.
