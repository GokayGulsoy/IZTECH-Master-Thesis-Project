#!/usr/bin/env bash
# H100 SXM Pod sweep — HyPER-LPAN cross-architecture validation.
#
# Runs MCKP composition selection then full HyPER-LPAN training for every
# (backbone, task, seed) combination. Designed for a fresh H100 SXM5 80GB
# RunPod instance with the repo cloned at /workspace.
#
# Expected wall-time on H100 SXM:
#   - SST-2 / QNLI: ~25-30 min/run  (large datasets)
#   - MRPC / RTE:   ~15-20 min/run  (small datasets)
#   Total: ~36 runs ≈ 18-22 hr. Budget ~$70-90 at $4/hr.
#
# Each run writes:
#   logs/sweep/{backbone}_{task}_seed{seed}.log
#   results/multi_model/{task}/{backbone}/hyper_lpan/  (final checkpoint + metrics)
#   results/composition/plan_mckp_{backbone}_{task}.json (one per backbone+task)
#
# Acceptance gate per task (reported in the post-run summary):
#   SST-2 ≥ 90%, MRPC F1 ≥ 86%, QNLI ≥ 88%, RTE ≥ 65% (median over 3 seeds)
#
# Usage on Pod:
#   git clone <repo> && cd Iztech_Master_Thesis_Implementation
#   python -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#   bash scripts/h100_sweep.sh                   # full sweep
#   bash scripts/h100_sweep.sh --dry-run         # print commands only
#   BACKBONES="base" TASKS="mrpc" bash scripts/h100_sweep.sh   # single combo
set -euo pipefail

# ── Configuration (overridable via env) ─────────────────────────────────────
BACKBONES="${BACKBONES:-base roberta-base distilbert}"
TASKS="${TASKS:-sst2 mrpc qnli rte}"
SEEDS="${SEEDS:-42 123 2024}"

BUDGET_FRACTION="${BUDGET_FRACTION:-0.67}"
MIN_LPAN="${MIN_LPAN:-4}"
GAMMA_Q="${GAMMA_Q:-0.5}"
N_SAMPLES="${N_SAMPLES:-256}"
COST_SCALE="${COST_SCALE:-10}"

# Pareto budget sweep. If empty, falls back to the single BUDGET_FRACTION above.
# Set e.g. BUDGETS="0.40 0.50 0.67 0.80 1.00" to produce per-task Pareto curves.
BUDGETS="${BUDGETS:-}"
if [[ -z "$BUDGETS" ]]; then BUDGETS="$BUDGET_FRACTION"; fi

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/sweep results/composition

# ── 0. Pre-flight ───────────────────────────────────────────────────────────
echo "=========================================================="
echo "  HyPER-LPAN H100 SXM Sweep"
echo "  backbones: $BACKBONES"
echo "  tasks:     $TASKS"
echo "  seeds:     $SEEDS"
echo "  budgets:   $BUDGETS  (min_lpan=$MIN_LPAN  gamma_q=$GAMMA_Q)"
echo "=========================================================="

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    eval "$@"
  fi
}

if [[ $DRY_RUN -eq 0 ]]; then
  # Sanity: GPU + bf16
  python -c "
import torch, sys
assert torch.cuda.is_available(), 'No CUDA device.'
n = torch.cuda.get_device_name(0)
bf16 = torch.cuda.is_bf16_supported()
print(f'  GPU: {n}  bf16: {bf16}')
sys.exit(0 if bf16 else 1)
"
fi

# ── 1. MCKP composition for every (backbone, task, budget) ────────────────
echo ""
echo "[Phase 1] MCKP composition selection"
for backbone in $BACKBONES; do
  for task in $TASKS; do
    for budget in $BUDGETS; do
      btag=$(printf 'b%02d' $(python -c "print(int(round(float('$budget')*100)))"))
      plan_file="results/composition/plan_mckp_${backbone}_${task}_${btag}.json"
      if [[ -f "$plan_file" && $DRY_RUN -eq 0 ]]; then
        echo "  skip ${backbone}/${task}/${btag} (plan exists)"
        continue
      fi
      ckpt_arg=""
      ft="results/multi_model/${task}/${backbone}/baseline/best_model"
      [[ -d "$ft" ]] && ckpt_arg="--checkpoint $ft"
      run "python experiments/select_composition.py \
        --model $backbone --task $task --method mckp \
        $ckpt_arg \
        --n-samples $N_SAMPLES --max-seq-len 128 \
        --budget-fraction $budget --min-lpan $MIN_LPAN \
        --gamma-q $GAMMA_Q --cost-scale $COST_SCALE \
        --out $plan_file \
        2>&1 | tee logs/sweep/select_${backbone}_${task}_${btag}.log"
    done
  done
done

# ── 2. Train every (backbone, task, budget, seed) ─────────────────────────
echo ""
echo "[Phase 2] HyPER-LPAN training"
for backbone in $BACKBONES; do
  for task in $TASKS; do
    base_cfg="configs/hyper_lpan/${task}_base.yaml"
    [[ ! -f "$base_cfg" ]] && { echo "  WARN: no config $base_cfg"; continue; }
    for budget in $BUDGETS; do
      btag=$(printf 'b%02d' $(python -c "print(int(round(float('$budget')*100)))"))
      plan_file="results/composition/plan_mckp_${backbone}_${task}_${btag}.json"
      if [[ ! -f "$plan_file" && $DRY_RUN -eq 0 ]]; then
        echo "  WARN: no plan $plan_file, skipping"; continue
      fi
      if [[ $DRY_RUN -eq 0 ]]; then
        lm=$(python -c "import json; p=json.load(open('$plan_file')); print(','.join(map(str, p['linear_mixing_layers'])))")
        qd=$(python -c "import json; p=json.load(open('$plan_file')); print(','.join(map(str, p['quad_attention_layers'])))")
      else
        lm="<from-plan>"; qd="<from-plan>"
      fi
      for seed in $SEEDS; do
        log="logs/sweep/${backbone}_${task}_${btag}_seed${seed}.log"
        out_dir="results/multi_model/${task}/${backbone}/hyper_lpan_${btag}_seed${seed}"
        if [[ -f "$out_dir/results.json" && $DRY_RUN -eq 0 ]]; then
          echo "  skip ${backbone}/${task}/${btag}/seed${seed} (already finished)"
          continue
        fi
        lm_arg=""; [[ -n "$lm" ]] && lm_arg="--linear-mixing-layers $lm"
        qd_arg=""; [[ -n "$qd" ]] && qd_arg="--quad-attention-layers $qd"
        run "python experiments/train_hyper_lpan.py \
          --config $base_cfg \
          --model $backbone \
          --seed $seed \
          $lm_arg $qd_arg \
          --output-dir $out_dir \
          2>&1 | tee $log"
      done
    done
  done
done

# ── 3. Summary ──────────────────────────────────────────────────────────────
echo ""
echo "[Phase 3] Sweep summary"
if [[ $DRY_RUN -eq 0 ]]; then
  python -c "
import json, glob, statistics, re
from pathlib import Path

results = {}
for d in glob.glob('results/multi_model/*/*/hyper_lpan_b*_seed*'):
    p = Path(d)
    m = re.match(r'hyper_lpan_(b\d+)_seed(\d+)', p.name)
    if not m: continue
    btag, seed = m.group(1), m.group(2)
    backbone = p.parent.name
    task = p.parent.parent.name
    rj = p / 'results.json'
    if not rj.exists(): continue
    s = json.loads(rj.read_text())
    final = s.get('final_accuracy')
    if final is None: continue
    results.setdefault((backbone, task, btag), []).append(float(final))

print(f'{\"backbone\":<14} {\"task\":<6} {\"budget\":<6} {\"n\":>2} {\"median\":>8} {\"min\":>8} {\"max\":>8}')
gates = {'sst2': 0.90, 'mrpc': 0.86, 'qnli': 0.88, 'rte': 0.65}
for (bb, t, btag), vals in sorted(results.items()):
    med = statistics.median(vals)
    mark = ' OK' if med >= gates.get(t, 0) else ' FAIL'
    print(f'{bb:<14} {t:<6} {btag:<6} {len(vals):>2} {med:>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}{mark}')
"
fi

echo ""
echo "Done. Logs in logs/sweep/, checkpoints in results/multi_model/."
echo "Generate Pareto figure with: python experiments/plot_pareto.py"
