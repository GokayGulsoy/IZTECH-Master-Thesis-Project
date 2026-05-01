# 05 — Reproducing Results

All commands assume `cwd = /home/gokay/Desktop/Iztech_Master_Thesis_Implementation`
and an active venv:

```bash
source fhe_venv/bin/activate
export PYTHONPATH=.
```

## 1. Train the unified HyPER-LPAN pipeline

Single command per (model, task). Resumable: re-running skips finished
stages via `.done` markers.

```bash
python experiments/train_hyper_lpan.py \
    --config configs/hyper_lpan/sst2_base.yaml \
    --device cuda
```

Configs available:
- `configs/hyper_lpan/sst2_base.yaml`
- `configs/hyper_lpan/mrpc_base.yaml`  (now `linear_mixing_layers: []`, `quad_attention_layers: [0..7]`)
- `configs/hyper_lpan/qnli_base.yaml`
- `configs/hyper_lpan/rte_base.yaml`

Outputs land in `results/multi_model/<task>/<model>/`:
- `lpan_staged_final/best_model/`
- `quad_progressive/best_model/`
- `linear_mixing_progressive/best_model/`
- `hybrid_progressive/best_model/`   ← final HyPER-LPAN checkpoint

## 2. Pick a task-adaptive composition (Ext 3)

```bash
python experiments/select_composition.py \
    --model base --task mrpc \
    --checkpoint results/multi_model/mrpc/base/lpan_staged_final/best_model \
    --n-samples 64 --max-seq-len 64 \
    --budget-fraction 0.5 --min-lpan 2
```

Writes `results/composition/plan_base_mrpc.json` and prints a YAML
fragment to paste into `configs/hyper_lpan/mrpc_base.yaml`.

## 3. Run encrypted inference benchmark (FHE)

Requires OpenFHE 1.2.3 + HEXL installed. On the Pod:

```bash
python experiments/run_fhe_benchmark.py \
    --model base --task sst2 \
    --hybrid \
    --checkpoint results/multi_model/sst2/base/hybrid_progressive/best_model \
    --linear-mixing-layers 0,1,2,3 \
    --quad-attention-layers 4,5,6,7 \
    --reduced-degrees \
    --max-seq-len 64 \
    --n-jobs 32 \
    --n-samples 100 \
    --word-elimination padding \
    --out results/benchmarks/
```

Writes `results/benchmarks/fhe_benchmark_base_sst2.json` with:
- `plaintext_accuracy`, `fhe_accuracy`, `agreement`
- `mean_latency_s`, `median_latency_s`, per-op timings
- `keygen_time_s` (one-off)

CLI flags worth knowing:
- `--word-elimination {none|padding|content_teacher}` (Ext W)
- `--keep-ratio 0.5` (content_teacher only)
- `--reduced-degrees` (Phase 2b polynomial degree pruning)
- `--no-bootstrap` (faster setup, limits depth — for quick smoke tests)
- `--dry-run` (skip encryption, just plaintext accuracy check)

## 4. Local FHE smoke test (no OpenFHE)

```bash
python experiments/run_fhe_benchmark.py \
    --model base --task sst2 --hybrid --dry-run
```

Confirms checkpoint loads + plaintext accuracy is reasonable before
spending Pod time.

## 5. Bootstrap schedule analysis

```python
from fhe_thesis.encryption.bootstrap_scheduler import (
    schedule_bootstraps, schedule_uniform, compare_plans,
    composition_to_kinds,
)
kinds = composition_to_kinds(12, lm_layers=[0,1,2,3], q_layers=[4,5,6,7])
print(compare_plans(kinds, budget_per_window=80, uniform_period=2))
```

## 6. Pareto plot (TODO)

`experiments/plot_pareto.py` — to be written. Will:
1. Glob `results/benchmarks/*.json`
2. Extract `(mean_latency_s, fhe_accuracy)` per run
3. Plot per-task Pareto front

## Common pitfalls

- **`ModuleNotFoundError: fhe_thesis`** → forgot `export PYTHONPATH=.`
- **`RuntimeError: CUDA out of memory`** → drop training batch size to
  8 and add `gradient_accumulation_steps: 2` in the YAML
- **NaN distillation loss** → a known L1 episode on MRPC v1; mitigated
  by lower LR (2e-5 → 1e-5) on Stage B for that layer, or use bf16 on H100
- **`apply_elimination` returns wrong shape** → caller forgot to also
  truncate `attention_mask` to the same `max_seq_len` before passing in
- **Bootstrap scheduler raises "Layer X has depth > budget"** → the
  per-window budget is too tight for any single layer; raise it or
  apply Phase 2b polynomial degree pruning

## End-to-end sanity check (one paste)

```bash
source fhe_venv/bin/activate && export PYTHONPATH=. && \
  python -c "
from fhe_thesis.encryption.elimination import apply_elimination
from fhe_thesis.optimization.composition_selector import select_composition
from fhe_thesis.encryption.bootstrap_scheduler import schedule_bootstraps, composition_to_kinds, LAYER_DEPTH
from fhe_thesis.poly.learnable import LearnablePolyAdapter
from fhe_thesis.models.backbone import get_encoder_layers
import numpy as np
print('imports OK')
print('LAYER_DEPTH:', LAYER_DEPTH)
"
```
