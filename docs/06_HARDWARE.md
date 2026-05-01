# 06 — Hardware

Three hardware roles in this project:

1. **Training** — distillation runs (LPAN baseline + 4 stages of HyPER-LPAN).
2. **FHE inference benchmarks** — wall-clock latency + memory profile.
3. **Plaintext analysis** — composition selector, plaintext-teacher
   eval, ablation sweeps.

## Available hardware

| Box | Specs | Where | Cost | Best for |
|---|---|---|---|---|
| **MSI laptop** | RTX 5070 Ti (12 GB VRAM), Threadripper-class CPU, 32 GB RAM | local | $0 | iteration, plaintext analysis, smoke tests |
| **RunPod H100 SXM** | H100 SXM5 (80 GB HBM3), bf16 tensor cores | cloud | ~$3.5–4/hr (community), ~$5–6/hr (secure) | full GLUE training sweeps |
| **Pod CPU instance** | Threadripper 7960X (32 vCPU), 256 GB RAM, AVX-512 + HEXL | cloud | ~$1.5–2/hr | FHE inference benchmarks (only place we run OpenFHE) |

## H100 vs RTX 5070 Ti — concrete numbers

For BERT-base + KD teacher (FP32 BERT-base in memory alongside student),
batch 16, seq 128, full HyPER-LPAN graph:

| Metric | RTX 5070 Ti | H100 SXM | H100 advantage |
|---|---|---|---|
| Peak VRAM used | ~10–11 GB (tight) | ~14 GB (trivial) | safety margin |
| FP32 throughput (BERT-base train) | ~0.6 K samples/s | ~3.5 K samples/s | **~6×** |
| BF16 throughput | not supported in our PyTorch / Blackwell path | ~7 K samples/s | **~12×** vs FP32 5070 Ti |
| SST-2 epoch (67 K samples, batch 16) | ~10 min | ~1.5 min (BF16) | ~7× |
| SST-2 full Stage A (3 epochs) | ~30 min | ~5 min | |
| Full HyPER-LPAN Stage A→D for one task | ~3–4 h | ~30–45 min | ~5× |
| Full sweep: 4 tasks × 3 backbones × 3 seeds | ~24–36 h | ~3–5 h | |

## Cost / time trade

For the full sweep (your primary deliverable):

| Plan | Wall time | Cash cost | Iteration friction |
|---|---|---|---|
| All on 5070 Ti | 24–36 h overnight × 1–2 nights | $0 | zero |
| All on H100 SXM | 3–5 h | ~$15–25 | upload checkpoints back |
| Hybrid: dev on 5070 Ti, final on H100 | 6–10 h | ~$25–40 | some |

**Updated recommendation (per your preference for speed): use H100 SXM
for the full sweep.** Reasons:

- **~6× speedup is "considerable"** — 24h → 4h is the difference
  between "single-night turnaround" and "results before lunch"
- ~$15–25 total cost for a full sweep is well within thesis budget
- BF16 on H100 (vs FP32 forced on Blackwell 5070 Ti's bf16 path under
  current PyTorch) gives an additional 2× cushion against any
  numerical-instability episode (we saw NaN on MRPC L1 in v1)
- Results carry back as a single tarball of `results/multi_model/`

### Suggested workflow

1. **Locally on 5070 Ti** (free, fast iteration):
   - Run `select_composition.py` for each task to derive selector plans
   - Smoke-test the unified pipeline with `--n-samples 256` to confirm
     configs/configs syntax
   - Validate Ext W / Ext 1 / Ext 2 wiring with a 1-epoch run per task

2. **On H100 SXM** (final training):
   - Upload `configs/`, `fhe_thesis/`, `experiments/`, `run_staged_lpan.py`
   - Stream GLUE datasets from HuggingFace cache (no large data upload)
   - Run all 4 tasks × 3 backbones × 3 seeds in one background job
   - Download `results/multi_model/` tarball when done

3. **On Pod CPU** (FHE benchmarks only — never train here):
   - Upload trained checkpoints
   - Run `experiments/run_fhe_benchmark.py` for each (task, model,
     extension config)
   - Download `results/benchmarks/` JSONs

### When to skip H100 and stay local

- Quick smoke tests, individual stage debugging
- Composition-selector runs (need only plaintext attentions)
- Anything that takes < 30 minutes locally
- Iterating on a config that's failing — H100 startup + checkpoint
  upload time dominates short jobs

## Pod (32-vCPU Threadripper 7960X) — FHE inference

This is the **only** place we run OpenFHE + HEXL. Required because:
- AVX-512 is mandatory for HEXL's NTT speedup (3× over reference NTT)
- 32 cores let us parallelise across tokens (the dominant axis with
  token-packed layout)
- 256 GB RAM is plenty for ciphertext buffers (one BERT-base inference
  uses ~8–12 GB ciphertext memory at depth 25)

### Latency target derivation (5–7 s/sample)

Starting from naive baseline ~400 s/sample (single-thread, no HEXL):

| Optimisation | Factor | Cumulative |
|---|---|---|
| Baseline | — | 400 s |
| 32-vCPU token parallelism | ÷ 16 (Amdahl-limited from 32) | 25 s |
| AVX-512 + HEXL NTT | ÷ 3 | 8.3 s |
| ct×ct reduction (HyPER-LPAN composition) | ÷ 1.5 | 5.5 s |
| Bootstrap reduction (Ext 1) | ÷ 1.1 | 5.0 s |
| Polynomial degree pruning (Phase 2b) | ÷ 1.05 | 4.8 s |

Target hit. Range with all extensions enabled: **5–7 s for
SST-2/RTE; 7–9 s for MRPC** (paraphrase needs more LPAN layers).

### Per-composition projections

| Composition | LM/Q/L | Bootstraps | Pod wall-time |
|---|---|---|---|
| Pure LPAN | 0/0/12 | 5 | ~20 s |
| HyPER-LPAN canonical | 4/4/4 | 4 | ~11 s |
| **SST-2 selector + Ext W padding** | 10/0/2 | 2 | **~5 s** |
| **MRPC selector + Ext W padding** | 4/4/4 | 4 | **~9 s** |
| **MRPC selector + Ext W content_teacher@0.5** | 4/4/4 | 4 | **~6 s** |
| **QNLI/RTE selector + Ext W padding** | 6/4/2 | 3 | **~7 s** |

### vs published numbers

| System | Hardware | Latency/sample | Threat model |
|---|---|---|---|
| **HyPER-LPAN** (us, projected) | 32-vCPU CPU + HEXL | **5–9 s** | pure FHE |
| BOLT | GPU + 2PC | 10.9 / 12.0 s | FHE+MPC |
| Iron | GPU + 2PC | 8.5 s | FHE+MPC |
| NEXUS | GPU + CKKS | 7.0 s | pure FHE |
| MPCFormer | GPU + 3PC | 5–10 s | secret sharing |

**Competitive with NEXUS** on a stronger hardware substrate (CPU vs
GPU) and with extra extensions (Ext W, Ext 3) that NEXUS lacks.

## Storage

- **Checkpoints**: `results/multi_model/` is gitignored (large, .safetensors).
  Tarball + transfer between MSI ↔ H100 ↔ Pod manually.
- **Configs**: `configs/` is in git — single source of truth.
- **Benchmarks**: `results/benchmarks/*.json` — small, can be committed
  if you want a reproducibility trail, otherwise gitignored with the
  rest of `results/`.
