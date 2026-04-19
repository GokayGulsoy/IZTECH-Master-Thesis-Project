# IZTECH Master Thesis: Privacy-Preserving Transformer Inference via Learnable Polynomial Activation Networks (LPAN)

**Author:** Gokay Gulsoy  
**Institution:** Izmir Institute of Technology (IZTECH)  
**Degree:** M.Sc. in Computer Engineering

## Overview

This repository contains the implementation and thesis document for privacy-preserving transformer inference using **Fully Homomorphic Encryption (FHE)**. The core contribution is the **LPAN (Learnable Polynomial Activation Network)** framework, which replaces all non-polynomial operations in BERT models (GELU, Softmax, LayerNorm) with trainable polynomial approximations, enabling encrypted inference under the CKKS scheme.

## What Has Been Done

### LPAN Framework
- **3-stage progressive polynomial replacement pipeline:**
  - **Stage 1:** Replace GELU activations with learnable polynomial approximations (cross-entropy fine-tuning)
  - **Stage 2:** Replace Softmax with per-head polynomial approximations (progressive layer-by-layer with Attention Knowledge Distillation)
  - **Stage 3:** Replace LayerNorm with polynomial approximations (progressive layer-by-layer with AttnKD + co-adaptation)
- **Weighted minimax polynomial approximation** using profiled activation distributions (3.4× error reduction vs standard minimax)
- **AttentionDistillationTrainer** with combined loss: `L = α·L_CE + β·Σ KL(A_teacher||A_student) + γ·Σ MSE(H_teacher, H_student)`
- **NaN-safe training** with gradient clipping, coefficient clamping, and automatic recovery

### Results on SST-2 (Sentiment Analysis)

| Model | Original | LPAN (All-Poly) | Accuracy Drop |
|-------|----------|-----------------|---------------|
| BERT-Tiny (2L/128H) | 83.26% | 83.14% | 0.11% |
| BERT-Mini (4L/256H) | 87.16% | 86.81% | 0.34% |
| BERT-Small (4L/512H) | 87.73% | 88.53% | +0.80% (gain) |
| BERT-Base (12L/768H) | 92.20% | 91.86% | 0.34% |

### Codebase
- `run_staged_lpan.py` — Main LPAN training orchestration (3-stage pipeline with resume support)
- `fhe_thesis/models/activations.py` — Learnable polynomial modules (GELU, Softmax, LayerNorm)
- `fhe_thesis/models/replacement.py` — Model surgery to inject polynomial modules into BERT
- `fhe_thesis/models/profiling.py` — Hook-based activation profiling + weighted minimax fitting
- `fhe_thesis/training/trainer.py` — NaNSafeTrainer, DistillationTrainer, AttentionDistillationTrainer
- `fhe_thesis/poly/approximation.py` — Weighted minimax, Taylor, Chebyshev, least-squares approximations
- `fhe_thesis/poly/chebyshev.py` — Clenshaw recurrence evaluation
- `extract_coefficients.py` — Extract trained polynomial coefficients from checkpoints
- `experiments/` — Individual experiment scripts (profiling, approximation, encrypted inference, etc.)

### Thesis Document
- Complete LaTeX thesis in `IZTECH_Master_Thesis/` with all 5 chapters, frontmatter, and references

## What Will Be Done (Planned Extensions)

### 1. Multi-Dataset Evaluation (GLUE Benchmark Extension)
- Extend LPAN evaluation beyond SST-2 to the GLUE tasks reported by all
  comparable FHE/MPC transformer works (MPCFormer, THE-X, BOLT):
  - **MRPC** (Microsoft Research Paraphrase Corpus)
  - **QNLI** (Question Natural Language Inference)
  - **QQP** (Quora Question Pairs)
- MNLI / RTE / STS-B are intentionally **out of scope** — they are not
  reported by the baseline papers, so adding them would not improve the
  comparison table.

#### Running on a new task

The task registry lives in [fhe_thesis/tasks.py](fhe_thesis/tasks.py)
(`GLUE_TASKS = {"sst2", "mrpc", "qnli", "qqp"}`). Both the baseline
trainer and the staged LPAN pipeline accept a `--task` flag that
threads task-specific metric, num_labels, and dataset loader through
every stage.

```bash
# 1. Train baselines + all-poly LPAN (single-shot) for one task
python experiments/05_multi_model_scaling.py \
    --models tiny mini small base \
    --task mrpc          # or qnli / qqp / sst2

# 2. Run the staged 3-stage LPAN pipeline (CE → Softmax KD → LN KD)
python run_staged_lpan.py --model tiny --task mrpc
python run_staged_lpan.py --model mini --task qnli
python run_staged_lpan.py --model small --task qqp
python run_staged_lpan.py --model base --task qqp
```

**Result paths.** SST-2 keeps the legacy layout
(`results/multi_model/<model>/...`) so existing checkpoints are not
disturbed. New tasks land under
`results/multi_model/<model>/<task>/...`. The summary JSON gets a
`_<task>` suffix when not `sst2`.

**Metric per task** (auto-selected by `compute_metrics_for_task` and
used as `metric_for_best_model`):

| Task | Primary metric | Also reported |
|------|----------------|---------------|
| sst2 | accuracy       | —             |
| mrpc | f1             | accuracy      |
| qnli | accuracy       | —             |
| qqp  | f1             | accuracy      |

### 2. Encrypted Inference with CKKS
- Complete end-to-end encrypted inference pipeline using the CKKS homomorphic encryption scheme
- Measure **encryption/decryption overhead**, **ciphertext computation latency**, and **memory usage**
- Evaluate latency breakdown: polynomial evaluation vs. linear layers vs. bootstrapping
- Compare plaintext vs. encrypted accuracy to quantify precision loss from CKKS noise

### 3. Latency and Performance Metrics
- Measure and visualize:
  - **End-to-end inference latency** (plaintext vs. encrypted)
  - **Per-layer latency breakdown** (attention, FFN, normalization)
  - **Multiplicative depth consumption** per polynomial operation
  - **Communication overhead** for client-server deployment
  - **Throughput** (sentences/second) under encryption
- Generate comparison figures against existing FHE-based transformer methods

### 4. Figure Regeneration
- Regenerate all experimental figures with final results:
  - Activation distribution plots (GELU, Softmax, LayerNorm inputs)
  - Polynomial approximation quality comparisons
  - Accuracy vs. polynomial degree trade-off curves
  - Encrypted vs. plaintext latency bar charts
  - Layer-by-layer accuracy trajectory during progressive replacement
  - SOTA comparison charts

### 5. Custom Privacy-Preserving Inference Protocol Design
- Design a unique end-to-end protocol for private transformer inference:
  - Client-side encryption of input tokens
  - Server-side homomorphic computation with LPAN-optimized polynomials
  - Optimized ciphertext packing strategy for batched attention computation
  - Bootstrapping scheduling to minimize depth consumption
  - Protocol specification with security guarantees and complexity analysis
- Formal analysis of multiplicative depth budget and parameter selection

### 6. Additional Improvements
- Automated polynomial degree selection based on accuracy-depth trade-off
- Support for larger architectures (BERT-Large, DistilBERT)
- GPU-accelerated CKKS operations for practical deployment
- Formal circuit depth analysis for each BERT variant

## CKKS Encrypted Inference (`feature/ckks-protocol`)

The CKKS work is being landed in incremental, runnable phases on the
`feature/ckks-protocol` branch. The full design is in
[docs/ckks_protocol.md](docs/ckks_protocol.md); a short operator-level
summary follows.

### Protocol — Pure-FHE Single-Round (PF-SR)

Existing FHE/MPC transformer systems (THE-X, MPCFormer, BOLT, Iron)
fall back to MPC round-trips for LayerNorm because `(x − μ)/σ`
requires a square root that is impractical under FHE. LPAN replaces
LN with a polynomial during training, so the **server can run the
entire transformer layer purely under FHE**:

```
Client            Server
──────            ──────
Enc(x) ──────────▶
                  LPAN_layer(ct)   ← no client interaction
       ◀────────  Enc(y)
Dec(y)
```

One round-trip total, independent of model depth.

### Module layout

```
fhe_thesis/encryption/
  context.py       # CKKS context factories (existing)
  backend.py       # CKKSBackend ABC + TenSEALBackend reference impl
  packing.py       # TokenPackedTensor — token-packed slot layout
  ops.py           # enc_linear, enc_gelu_poly, enc_ln_poly,
                   # enc_qk_scores, enc_softmax_poly,
                   # enc_attention_apply, enc_self_attention
  coefficients.py  # PolyCoeffs + load_coefficients(model_key)
                   #   reads results/coefficients/bert_<model>_coeffs.json
                   #   (LPAN-trained), falls back to profile-and-fit
  protocol.py      # model-agnostic encrypted blocks:
                   #   encrypt_ffn_block / encrypt_attention_block /
                   #   encrypt_layer / encrypt_inference / run_phase
  depth.py         # symbolic depth audit (DepthAudit, DEPTH_COST)
docs/
  ckks_protocol.md   # full protocol specification
experiments/
  run_protocol.py    # unified CLI: --model {tiny|mini|small|base}
                     #              --phase {ffn|attention|layer|model}
```

The model-agnostic protocol pulls `hidden`, `heads`, `layers` from
`MODEL_REGISTRY` so the **same code path runs on Tiny / Mini / Small /
Base** — no per-model branching.

### Packing strategy (token-packed)

One ciphertext per token row, `hidden_dim` slots used per ciphertext.
Chosen because every LPAN polynomial (GELU, softmax-poly, LN-poly) is
**intra-token element-wise**, so polynomial evaluation costs **zero
rotations**. Linear layers use a per-row plaintext-weight matmul.

### Multiplicative-depth budget

Critical-path depth per LPAN BERT layer (Q/K/V parallel, two LN-polys
sequential, degree-8 polys via Horner, multi-head concat under FHE) =
**23 levels**. Computed by
`fhe_thesis.encryption.depth.transformer_layer_depth()`. The Phase-1
FFN+LN block alone is 9 levels and fits a TenSEAL N=16384 chain
without bootstrapping; full layers / multi-layer models need
N=32768 or bootstrapping.

### Running on the MSI box

Same single command for every variant and every phase:

```bash
git fetch && git checkout feature/ckks-protocol
# 1) (optional) extract LPAN-trained coefficients
python extract_coefficients.py
# 2) run any phase × model combination
python experiments/run_protocol.py --model tiny --phase ffn
python experiments/run_protocol.py --model tiny --phase attention
python experiments/run_protocol.py --model mini --phase layer
python experiments/run_protocol.py --model base --phase model
```

Outputs land in `results/encrypted_inference/<model>_<phase>.json`
with per-step latency, output norm, and depth-budget summary.

### Roadmap

| Phase | Deliverable |
|---|---|
| 1 ✅ | Protocol design, backend abstraction, FFN+LN block |
| 2 ✅ | Encrypted multi-head self-attention |
| 3 ✅ | Model-agnostic full layer + classifier head, unified CLI |
| 4 | Scaling benchmark across Tiny / Mini / Small / Base |
| 5 | GPU-backend port (Phantom-FHE or OpenFHE-CUDA) — perf-only |

### Notes for the MSI box

- The protocol layer is intentionally backend-agnostic. Swapping
  TenSEAL for a GPU CKKS library is a single new `CKKSBackend`
  subclass — no protocol code changes.
- `fhe_thesis/encryption/__init__.py` uses lazy (PEP-562) imports so
  the design machine can load `depth` and `packing` even without
  `tenseal` installed; importing `TenSEALBackend` / `protocol`
  triggers the heavy import only when needed.
- Bootstrapping is intentionally **not** in scope yet. If a future
  phase needs it for BERT-Base, it will be added behind the
  `BackendCapabilities.supports_bootstrapping` flag.

## Environment

- **Hardware:** NVIDIA RTX 5070 Ti Laptop GPU (12GB VRAM)
- **Python:** 3.10 (venv at `fhe_venv/`)
- **Key dependencies:** PyTorch, Transformers (HuggingFace), Brevitas, TenSEAL, Datasets

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
