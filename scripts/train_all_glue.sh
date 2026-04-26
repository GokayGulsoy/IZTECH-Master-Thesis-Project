#!/usr/bin/env bash
# Train all (model × task) combinations: 4 models × 4 GLUE tasks.
#
# For each (model, task) pair we run:
#   1. baseline + poly-zs/poly-FT   (experiments/05_multi_model_scaling.py)
#   2. 3-stage LPAN curriculum      (run_staged_lpan.py)
#
# The script is idempotent: if a final LPAN checkpoint already exists for
# a (model, task), that combo is skipped. Logs are written per-combo to
# results/multi_model/<model>/<task>/run.log (sst2 → results/multi_model/<model>/run.log
# for backwards compatibility with the legacy directory layout).
#
# Usage:
#   bash scripts/train_all_glue.sh                       # all 4×4
#   MODELS="tiny mini" TASKS="mrpc qnli" bash scripts/train_all_glue.sh
#   DRY_RUN=1 bash scripts/train_all_glue.sh             # just print plan

set -u  # unset vars are errors; do NOT set -e (we want to keep going on failure)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use the project virtualenv's python explicitly so this script also
# works when launched from a non-activated shell (e.g. via nohup).
PYTHON="${PYTHON:-$ROOT/fhe_venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python interpreter not found at $PYTHON"
    exit 1
fi

# Make `import fhe_thesis` work even when launched from a non-activated
# shell (e.g. via nohup) where the venv's site-packages is on PATH but
# the project root is not on PYTHONPATH.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODELS="${MODELS:-tiny mini small base}"
TASKS="${TASKS:-sst2 mrpc qnli qqp}"
DRY_RUN="${DRY_RUN:-0}"

task_subdir() {
    # SST-2 keeps the legacy flat layout; other tasks live in a subdir.
    local task="$1"
    if [[ "$task" == "sst2" ]]; then echo ""; else echo "$task/"; fi
}

run_combo() {
    local model="$1" task="$2"
    local sub log_dir log_file final_ckpt baseline_ckpt
    sub="$(task_subdir "$task")"
    log_dir="results/multi_model/${model}/${sub%/}"
    [[ -z "$sub" ]] && log_dir="results/multi_model/${model}"
    log_file="${log_dir}/run.log"
    baseline_ckpt="results/multi_model/${model}/${sub}baseline/best_model"
    final_ckpt="results/multi_model/${model}/${sub}staged_lpan_final/best_model"

    if [[ -d "$final_ckpt" ]]; then
        echo "[skip] $model/$task — already complete ($final_ckpt)"
        return 0
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[plan] $model/$task → $log_file"
        return 0
    fi

    mkdir -p "$log_dir"
    {
        echo "================================================================"
        echo "  $(date -Iseconds)  $model × $task"
        echo "================================================================"
    } | tee -a "$log_file"

    # ── Step 1: baseline finetune (only if missing) ──
    if [[ ! -d "$baseline_ckpt" ]]; then
        echo "[step 1/2] baseline finetune  $model × $task" | tee -a "$log_file"
        if ! "$PYTHON" experiments/05_multi_model_scaling.py \
                --models "$model" --task "$task" \
                >> "$log_file" 2>&1; then
            echo "[FAIL] baseline finetune $model × $task — see $log_file"
            return 1
        fi
    else
        echo "[step 1/2] baseline already present — skipping" | tee -a "$log_file"
    fi

    # ── Step 2: 3-stage LPAN ──
    echo "[step 2/2] staged-LPAN curriculum  $model × $task" | tee -a "$log_file"
    if ! "$PYTHON" run_staged_lpan.py \
            --model "$model" --task "$task" \
            >> "$log_file" 2>&1; then
        echo "[FAIL] staged-LPAN $model × $task — see $log_file"
        return 1
    fi

    echo "[done] $model × $task" | tee -a "$log_file"
    return 0
}

echo "================================================================"
echo "  GLUE training matrix   models='$MODELS'   tasks='$TASKS'"
echo "  python = $PYTHON"
echo "  dry run = $DRY_RUN"
echo "================================================================"

failed=()
for task in $TASKS; do
    for model in $MODELS; do
        if ! run_combo "$model" "$task"; then
            failed+=("$model/$task")
        fi
    done
done

echo
echo "================================================================"
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "  All combos OK"
    exit 0
else
    echo "  ${#failed[@]} combo(s) failed:"
    for f in "${failed[@]}"; do echo "    - $f"; done
    exit 1
fi
