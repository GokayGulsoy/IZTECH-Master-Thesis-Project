# 07 — Repository Layout

## Top-level

```
.
├── configs/                      # YAML configs (single source of truth)
│   └── hyper_lpan/
│       ├── sst2_base.yaml
│       ├── mrpc_base.yaml
│       ├── qnli_base.yaml
│       └── rte_base.yaml
├── docs/                         # this folder — design references
├── experiments/                  # CLI scripts (entry points)
│   ├── train_hyper_lpan.py       # unified training pipeline
│   ├── run_fhe_benchmark.py      # FHE inference benchmark
│   └── select_composition.py     # Ext 3 — pick layer assignment
├── fhe_thesis/                   # library
│   ├── benchmarks/               # accuracy + latency harness
│   ├── encryption/               # CKKS protocol + OpenFHE backend
│   ├── models/                   # LPAN / Quad / LinearMixing modules
│   ├── optimization/             # composition selector
│   ├── pipelines/                # HyperLPANPipeline orchestrator
│   ├── poly/                     # Chebyshev + learnable adapters
│   ├── training/                 # KD trainer, distillation losses
│   ├── config.py                 # MODEL_REGISTRY + paths + intervals
│   └── tasks.py                  # GLUE task metadata
├── results/                      # gitignored — checkpoints, JSONs
├── run_staged_lpan.py            # LPAN baseline (Stage A standalone)
├── run_stage4_range_aware.py     # optional Stage-4 range-aware FT
├── IZTECH_Master_Thesis/         # LaTeX thesis source
├── research_papers/              # PDFs of related work
├── fhe_venv/                     # gitignored — venv
└── .gitignore
```

## Module map (`fhe_thesis/`)

### `encryption/` — CKKS protocol layer

| File | Purpose |
|---|---|
| `protocol.py` | end-to-end `encrypt_inference[_hybrid|_linear_mixing]` + per-layer dispatch |
| `openfhe_backend.py` | OpenFHE 1.2.3 wrapper with EvalBootstrap |
| `backend.py` | abstract backend interface (allows plaintext-only stub) |
| `ops.py` | per-op primitives (mul_plain, add, etc.) |
| `packing.py` | TokenPackedTensor (one ct per token) |
| `coefficients.py` | load fixed Chebyshev coefficients |
| `hybrid_coefficients.py` | per-region coefficient loader (Phase 2b reduced degrees) |
| `depth.py` | symbolic CKKS-level cost per op + per layer kind |
| **`bootstrap_scheduler.py`** | **Ext 1 — region-adaptive bootstrap placement** |
| **`elimination.py`** | **Ext W — word/token elimination (padding + content_teacher)** |

### `models/` — PyTorch modules

| File | Purpose |
|---|---|
| **`backbone.py`** | **Ext 4 — cross-arch resolver (BERT/RoBERTa/DistilBERT)** |
| `lpan_loader.py` | `load_lpan_model(model_key, checkpoint_path)` |
| `linear_mixing.py` | `MultiHeadLinearMixingAttention` + `replace_with_linear_mixing` |
| `quad_attention.py` | `QuadAttention` (2Quad) + `replace_with_quad` |
| `hybrid_attention.py` | composition-aware replacement + `summarize_attention_types` |
| `replacement.py` | `replace_activations` (GELU/Softmax/LN → polynomial) |
| `polynomials.py` | `PolynomialGELU`, `PerHeadPolynomialSoftmax`, `PolynomialLayerNorm` |
| `profiling.py` | activation-distribution profiling for poly fits |

### `poly/`

| File | Purpose |
|---|---|
| `chebyshev.py` | Clenshaw recurrence (torch + numpy) |
| `approximation.py` | Remez / weighted minimax / Taylor / least-squares |
| **`learnable.py`** | **Ext 2 — `LearnablePolyAdapter` + `collect_fidelity_loss`** |

### `optimization/`

| File | Purpose |
|---|---|
| **`composition_selector.py`** | **Ext 3 — entropy-based composition planner** |

### `training/`

| File | Purpose |
|---|---|
| `trainer.py` | `attn_distill_and_eval` — KD + cross-entropy training loop |
| `distillation.py` | `AttnKDLoss` + helpers |
| `schedulers.py` | LR + KD-weight schedulers |
| `checkpoints.py` | resume support, safetensors loader |

### `pipelines/`

| File | Purpose |
|---|---|
| `hyper_lpan.py` | `HyperLPANPipeline` — orchestrates Stage A→D |
| `stages/` | one file per stage (A: LPAN, B: Quad, C: LinearMixing, D: Global FT) |

### `benchmarks/`

| File | Purpose |
|---|---|
| `accuracy.py` | GLUE eval harness, `evaluate_checkpoint`, `compare_checkpoints` |
| `latency.py` | `profile_latency`, `LatencyResult`, `aggregate_timings` |
| `__init__.py` | re-exports |

## Branch structure

```
main                              ← stable, public-facing
└── feature/ckks-protocol          ← validated baseline (commits 297d3f1, 14a43c2)
    └── feature/hyper-lpan-extensions  ← HEAD; 5 ext + cleanup
        d288662  feat(elim): word elimination
        71c25b9  feat(composition): task-adaptive selector
        6d9b249  feat(bootstrap): region-adaptive scheduler
        c9cd28d  feat(poly): learnable Chebyshev + range tracking
        4bdb02e  feat(arch): RoBERTa + DistilBERT support
        e3845dd  chore(cleanup): remove superseded finetune scripts
```

## File conventions

- **Configs**: YAML, snake_case keys, `<task>_<model>.yaml` naming
- **Checkpoints**: `results/multi_model/<task>/<model>/<stage_name>/best_model/`
  with `model.safetensors` + `config.json` + `tokenizer/`
- **Results JSONs**: `results/benchmarks/fhe_benchmark_<model>_<task>.json`
- **Composition plans**: `results/composition/plan_<model>_<task>.json`
- **Imports**: prefer `from fhe_thesis.X.Y import Z` over relative imports
- **CLI flags**: `--kebab-case`, e.g. `--word-elimination`, `--keep-ratio`

## Where to look first when something breaks

| Symptom | Look at |
|---|---|
| Training NaN loss | `fhe_thesis/training/trainer.py` + KD γ schedule in YAML |
| Wrong composition picked | `fhe_thesis/optimization/composition_selector.py` |
| FHE wall-time too high | `fhe_thesis/benchmarks/latency.py` per-op timings |
| Encrypted ≠ plaintext output | `fhe_thesis/encryption/protocol.py` + check polynomial fit ranges |
| OOM during training | drop `train_batch_size`, raise `gradient_accumulation_steps` in YAML |
| Bootstrap raises depth error | run `compare_plans()` to find a feasible budget |
| Cross-arch failure | `fhe_thesis/models/backbone.py` — add new entry to `_BACKBONE_PATHS` |
