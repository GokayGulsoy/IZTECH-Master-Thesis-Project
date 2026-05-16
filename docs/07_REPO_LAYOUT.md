# 07 — Repository Layout

Reflects the `synthesizer-lpan-production` branch (May 2026).

## Top-level

```
.
├── docs/                          # this folder — design references
│   ├── README.md
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── 01_ARCHITECTURE.md
│   ├── 02_FHE_PROTOCOL.md
│   ├── 03_THREAT_MODEL.md
│   ├── 04_OPTIMIZATIONS.md
│   ├── 05_REPRODUCING_RESULTS.md
│   ├── 06_HARDWARE.md
│   ├── 07_REPO_LAYOUT.md
│   └── TECHNIQUES_JOURNEY.md      # narrative of what was tried
├── fhe_thesis/                    # library
│   ├── __init__.py
│   ├── config.py                  # paths, MODEL_REGISTRY, intervals
│   ├── tasks.py                   # GLUE task metadata
│   ├── encryption/                # CKKS protocol + HEonGPU backend
│   │   ├── backend.py             # abstract CKKSBackend interface
│   │   ├── heongpu_backend.py     # CUDA HEonGPU wrapper
│   │   ├── heongpu_bindings/      # pybind11 sources + build.sh
│   │   ├── colmajor.py            # NEXUS column-major packing helpers
│   │   ├── multi.py               # multi-ciphertext bundle ops
│   │   ├── linear.py              # BSGS / streaming / multi linear projections
│   │   ├── layernorm.py           # cubic-invsqrt LayerNorm
│   │   └── attention.py           # Synthesizer attention (naive + BSGS)
│   ├── poly/                      # polynomial approximations
│   │   ├── approximation.py       # Remez / Chebyshev / weighted minimax
│   │   └── chebyshev.py           # Clenshaw recurrence
│   ├── models/                    # PyTorch modules + surgery
│   │   ├── activations.py         # PolynomialGELU, PolynomialLN, PolynomialSoftmax
│   │   ├── profiling.py           # hook-based activation profiling
│   │   ├── replacement.py         # surgery: inject polynomial activations / Synthesizer
│   │   └── backbone.py            # cross-arch resolver (BERT / RoBERTa / DistilBERT)
│   └── training/
│       ├── trainer.py             # KD + cross-entropy training loop
│       ├── checkpoints.py         # safetensors load/resume
│       ├── run_staged_lpan.py     # Stage-1 to Stage-3 LPAN CLI
│       ├── run_synth_lpan.py      # Stage-4 Synthesizer-LPAN CLI
│       └── export_synth_lpan.py   # bench JSON exporter
├── scripts/                       # CLI entry points
│   ├── setup_pod_gpu.sh           # one-shot HEonGPU build
│   ├── smoke_heongpu_backend.py   # backend sanity check
│   ├── test_synthesizer_lpan_correctness.py
│   └── bench_L128_synthesizer_lpan.py    # headline benchmark
├── third_party/
│   ├── HEonGPU/                   # vendored 8 MB
│   └── HEonGPU.commit             # pinned upstream commit
├── experiments/                   # thin wrappers around production CLIs
│   ├── run_staged_lpan.py
│   ├── run_synth_lpan_stage4.py
│   └── export_synth_lpan.py
├── results/                       # gitignored — benchmark JSONs, checkpoints
├── logs/                          # gitignored — training/bench logs
├── research_papers/               # PDFs of related work
├── IZTECH_Master_Thesis/          # LaTeX thesis source
├── fhe_venv/                      # gitignored — Python venv
├── README.md
├── pyproject.toml
├── LICENSE
└── .gitignore
```

## Module map (`fhe_thesis/`)

### `encryption/` — CKKS protocol layer

| File | Purpose |
|---|---|
| `backend.py` | Abstract `CKKSBackend` + `Ciphertext` types |
| `heongpu_backend.py` | HEonGPU CUDA implementation |
| `heongpu_bindings/` | pybind11 source + `build.sh` |
| `colmajor.py` | NEXUS column-major slot layout helpers + Galois key prep |
| `multi.py` | Multi-ciphertext bundle ops (cross-bundle add / rotate) |
| `linear.py` | Linear projections — BSGS, streaming, multi variants |
| `layernorm.py` | Cubic-invsqrt LayerNorm (single-ct + multi-bundle) |
| **`attention.py`** | **Synthesizer-LPAN attention — `attn_synthesizer`, `attn_synthesizer_bsgs`, `encode_synthesizer_diagonals`, `encode_synthesizer_bsgs`** |

### `poly/`

| File | Purpose |
|---|---|
| `approximation.py` | Remez, Chebyshev, weighted minimax, Taylor, least-squares |
| `chebyshev.py` | Clenshaw recurrence (torch + numpy) |

### `models/` — PyTorch modules

| File | Purpose |
|---|---|
| `activations.py` | `PolynomialGELU`, `PerHeadPolynomialSoftmax`, `PolynomialLayerNorm` |
| `profiling.py` | Hook-based activation distribution profiling for poly fits |
| `replacement.py` | `replace_activations`, `replace_attention_with_synthesizer` |
| `backbone.py` | Cross-arch resolver (BERT / RoBERTa / DistilBERT) |

### `training/`

| File | Purpose |
|---|---|
| `trainer.py` | KD + CE training loop, `attn_distill_and_eval`, `synth_attn_distill_and_eval` |
| `checkpoints.py` | Safetensors loader, resume support |
| `run_staged_lpan.py` | Stage-1 to Stage-3 LPAN teacher-chain training CLI |
| `run_synth_lpan.py` | Stage-4 Synthesizer-LPAN training CLI |
| `export_synth_lpan.py` | Export learned Synthesizer patterns + polynomial coeffs to bench JSON |

### Top-level

| File | Purpose |
|---|---|
| `config.py` | `MODEL_REGISTRY`, paths, polynomial intervals |
| `tasks.py` | GLUE task metadata (SST-2, MRPC, QNLI, QQP, RTE) |

## `scripts/` — CLI entry points

| File | Purpose |
|---|---|
| `setup_pod_gpu.sh` | One-shot HEonGPU build on stock Ubuntu + CUDA 12 |
| `smoke_heongpu_backend.py` | Backend import + basic ops sanity check |
| `test_synthesizer_lpan_correctness.py` | Naive vs BSGS vs plaintext equivalence |
| **`bench_L128_synthesizer_lpan.py`** | **Headline benchmark — now also accepts `--checkpoint bench_checkpoint.json`** |

## Branch structure

```
main                                   ← stable baseline
└── feature/ckks-protocol               ← validated CKKS protocol baseline
    └── feature/hyper-lpan-extensions   ← prior LPAN extensions (archived)
        └── synthesizer-lpan-production ← HEAD; current production branch
```

The production branch invariants:

- `third_party/HEonGPU/` vendored, commit pinned
- All public APIs free of `_nexus` / `_lpan_v2` / `hyper_` suffixes
- All modules under `fhe_thesis/encryption/` are focused and < 800 lines
- All scripts in `scripts/` import successfully and have a `--help`
- No dead code paths (no `multi_modal/`, no `composition/`, no
  `pipelines/` orchestrator legacy)

## File conventions

- **Imports**: prefer `from fhe_thesis.X.Y import Z` over relative
- **CLI flags**: `--kebab-case`
- **Benchmark JSONs**: `results/bench_<config>.json`
- **Checkpoints**: `results/synthesizer_lpan/<task>/<arch>/best_model/`
- **Bench exports**: `results/synthesizer_lpan/<task>/<arch>/bench_checkpoint.json`

## Where to look first when something breaks

| Symptom | Look at |
|---|---|
| HEonGPU import error | `scripts/setup_pod_gpu.sh` rerun + check CUDA driver |
| Numerical mismatch vs plaintext | `scripts/test_synthesizer_lpan_correctness.py` |
| FHE wall-time regression | `scripts/bench_L128_synthesizer_lpan.py` per-layer breakdown |
| OOM at BATCH=16 | drop to BATCH=8 + chain=24 |
| Chain=21 LN failure | expected; use chain=22 |
| Cross-arch failure | `fhe_thesis/models/backbone.py` `_BACKBONE_PATHS` table |
| Encrypted ≠ plaintext output | check polynomial input ranges in `fhe_thesis/poly/approximation.py` |
