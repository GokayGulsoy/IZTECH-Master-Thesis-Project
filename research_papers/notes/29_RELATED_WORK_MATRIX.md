# Related-Work Matrix — Claim-Safe Thesis Positioning

Purpose: give a thesis-safe comparison surface after the May 2026 literature re-pass.

## Claim-safe headline

What we can claim:

- **First FHE port of Synthesizer attention** in a BERT-style inference pipeline.
- **Architectural elimination** of $W_q$, $W_k$, $QK^\top$, and runtime Softmax from the encrypted attention path.
- **13.67x speedup** over the honest LPAN baseline on the same plain HEonGPU execution stack.
- **Single-GPU sub-100-second** end-to-end BERT result on plain HEonGPU.
- **Pure non-interactive FHE** threat model matching NEXUS and Cerium.

What we should not claim:

- Current single-GPU latency SOTA.
- Current overall pure-FHE latency SOTA.
- First sub-100-second pure-FHE BERT result without qualification.

Cerium changes the claim surface: it reports **36.1 s on 1x H100** and **8.8 s on 8x B200** for pure-FHE BERT-Base.

## Matrix

| Work | Threat model | Attention path | Main lever | How to position vs Synthesizer-LPAN |
|---|---|---|---|---|
| **Cerium (2025)** | pure FHE | standard attention | compiler/runtime + multi-GPU systems | Current latency leader; complementary system contribution, not our direct architectural contribution |
| **NEXUS (2025)** | pure FHE | standard attention | packing, compression, argmax, GPU protocol engineering | Closest same-threat-model baseline; system-level optimization around plain BERT |
| **BOLT (2024)** | interactive HE+2PC | standard attention | MPC protocols + approx-aware fine-tuning + word elimination | Strong MPC baseline, but weaker threat model |
| **Iron (2022)** | interactive HE+2PC | standard attention | foundational private-transformer protocol suite | Historical baseline, weaker threat model |
| **MPCFormer (2023)** | interactive MPC | standard attention | cheap approximations + KD | Training recipe prior art, not same crypto model |
| **THE-X (2022)** | weakened HE/client-aided | modified attention / client non-linears | cheap substitutions with leaked non-linear inputs | Early HE-NLP baseline with weaker privacy |
| **THEF (2024)** | HE+TEE | standard attention | enclave offload for non-linears | Weaker trust model; shows why pure-FHE-friendly architectures matter |
| **Synthesizer (2021)** | plaintext | synthetic fixed mixer | remove token-token interactions | Direct architectural parent; strongest justification for our mainline |
| **Linformer (2020)** | plaintext | low-rank standard attention | sequence projection | Bad fit for current CKKS packing; not worth pivoting to now |
| **Performer (2021)** | plaintext | kernelized query-conditioned attention | FAVOR+ random features | Still query dependent at inference; weaker FHE fit than Synthesizer |
| **Nyströmformer (2021)** | plaintext | landmark low-rank attention | Nyström approximation | Same issue: approximates attention rather than removing it |
| **AFT (2021)** | plaintext | query-conditioned mixer | attention-free gating | Near-neighbor but still query dependent at runtime |
| **FNet (2022)** | plaintext | fixed Fourier mixer | token mixing without attention | Best alternative future architecture branch; not a blocker before current training |
| **AutoFHE (2024)** | FHE CNNs | not Transformer attention | layer-wise mixed-degree + bootstrap search | Best prior art for search/heterogeneity methodology, not for attention architecture |
| **PoWER-BERT / LAT (2020/2021)** | plaintext | standard attention with token pruning | progressive length reduction | Useful for later compression experiments, but dynamic pruning is not the first branch to pursue |

## Decision outcome

1. Keep **Synthesizer-LPAN** as the main architecture for the thesis.
2. Do **not** pivot pre-pod to Linformer, Performer, Nyströmformer, or AFT.
3. The most promising near-term improvement is **pattern compression of the learned Synthesizer matrix**: diagonal pruning, band-limiting, or low-rank compression of exported $A_h$.
4. The strongest future alternative branch is **FNet**, because it is also a fixed, input-independent linear mixer under FHE.

## Ready-to-use thesis sentences

- "Cerium is the current pure-FHE latency leader, but it optimizes the standard Transformer stack via a compiler/runtime system; our contribution is orthogonal and architectural, removing the expensive encrypted attention path itself."
- "NEXUS is the closest prior work in our threat model; it keeps standard attention and optimizes packing and protocol mechanics, whereas Synthesizer-LPAN changes the model architecture to lower the encrypted circuit cost."
- "Among efficient-attention families, Synthesizer is uniquely well aligned with FHE because it can be frozen into a plaintext mixing matrix and therefore removes query-key interactions from the encrypted circuit entirely."
- "We therefore position Synthesizer-LPAN not as the current latency SOTA, but as the first FHE realization of Synthesizer-style attention and an architectural complement to modern FHE compilers."

## Training implication

Do not spend pod time on architecture pivots until the baseline Synthesizer-LPAN task sweep is complete. Spend the first follow-up budget on **real-checkpoint pattern compression ablations** instead.