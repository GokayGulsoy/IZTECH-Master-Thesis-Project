#!/usr/bin/env bash
# fill_matrix.sh — Sequential queue to fill the 16-cell GLUE x model matrix.
#
# Strategy:
#   1. Train Stage 1+2+3 (full LPAN pipeline) for every (model, task) cell
#      that lacks ``staged_lpan_final/best_model``.
#   2. Run Stage-4 range-aware refinement for every cell that has S3 but not S4.
#   3. Re-extract polynomial coefficients for cells that have a fresh Stage-4.
#   4. Run SPO end-to-end validation for every cell with extracted coefficients.
#
# Designed to be resumable: each step is idempotent — if the output exists, skip.
# Logs go to logs/matrix/<model>_<task>_<step>.log.
# Heartbeat written to logs/matrix/_status.txt so you know what's running.
#
# Usage:
#   bash fill_matrix.sh                # train all missing cells then validate
#   FORCE=1 bash fill_matrix.sh        # ignore existing outputs
#   STEPS="train" bash fill_matrix.sh  # only run training, skip validation
#   bash fill_matrix.sh status         # print progress and quit
#
# Resuming after a shutdown:
#   Just re-run `bash fill_matrix.sh` (or relaunch via nohup). It will
#   resume from the next missing output. No state file required.
#
set -u  # do not -e: a single cell failure shouldn't stop the queue

cd "$(dirname "$0")"
mkdir -p logs/matrix results/encrypted_inference

STATUS_FILE="logs/matrix/_status.txt"

note_status() {
    printf '[%s] %s\n' "$(date -Iseconds)" "$*" >> "$STATUS_FILE"
}

if [[ "${1:-}" == "status" ]]; then
    echo "===== matrix progress ====="
    for m in tiny mini small base; do
        for t in sst2 mrpc qnli qqp; do
            if [[ "$t" == "sst2" ]]; then
                s3="results/multi_model/$m/staged_lpan_final/best_model"
                s4="results/multi_model/$m/stage4_range_aware/best_model"
            else
                s3="results/multi_model/$m/$t/staged_lpan_final/best_model"
                s4="results/multi_model/$m/$t/stage4_range_aware/best_model"
            fi
            spo="results/encrypted_inference/${m}_${t}_spo.json"
            S3=$([[ -f "$s3/model.safetensors" ]] && echo "✅" || echo "❌")
            S4=$([[ -f "$s4/model.safetensors" ]] && echo "✅" || echo "❌")
            SP=$([[ -f "$spo" ]] && echo "✅" || echo "❌")
            printf "  %-6s %-5s  S3:%s  S4:%s  SPO:%s\n" "$m" "$t" "$S3" "$S4" "$SP"
        done
    done
    echo
    echo "===== last 20 status entries ====="
    tail -20 "$STATUS_FILE" 2>/dev/null || echo "(no status entries yet)"
    echo
    echo "===== current jobs ====="
    pgrep -af 'validate_e2e_spo|run_staged_lpan|run_stage4|extract_coefficients|05_multi_model' \
        || echo "  (none)"
    exit 0
fi

note_status "fill_matrix.sh started (PID=$$)"

PY="fhe_venv/bin/python"
export PYTHONPATH="."
export PYTHONUNBUFFERED="1"

MODELS=("tiny" "mini" "small" "base")
TASKS=("sst2" "mrpc" "qnli" "qqp")
STEPS="${STEPS:-train s4 extract validate}"
FORCE="${FORCE:-0}"

# ---------------------------------------------------------------------
# Helper: locate the Stage-3 final model dir for (model, task).
# ---------------------------------------------------------------------
s3_dir() {
    local m="$1" t="$2"
    if [[ "$t" == "sst2" ]]; then
        echo "results/multi_model/$m/staged_lpan_final/best_model"
    else
        echo "results/multi_model/$m/$t/staged_lpan_final/best_model"
    fi
}
s4_dir() {
    local m="$1" t="$2"
    if [[ "$t" == "sst2" ]]; then
        echo "results/multi_model/$m/stage4_range_aware/best_model"
    else
        echo "results/multi_model/$m/$t/stage4_range_aware/best_model"
    fi
}

# ---------------------------------------------------------------------
# Wait for any currently-running validation/training job to finish.
# We never run two heavy CPU jobs in parallel — they'd thrash.
# ---------------------------------------------------------------------
wait_idle() {
    while pgrep -f 'validate_e2e_spo|run_staged_lpan|run_stage4|05_multi_model' >/dev/null; do
        sleep 60
    done
}

# ---------------------------------------------------------------------
# 1. Stage 1+2+3 for missing cells.
# ---------------------------------------------------------------------
if [[ " $STEPS " == *" train "* ]]; then
  echo "===== STEP 1: Stage 1+2+3 training ====="
  for m in "${MODELS[@]}"; do
    for t in "${TASKS[@]}"; do
        s3="$(s3_dir "$m" "$t")"
        if [[ -f "$s3/model.safetensors" && "$FORCE" != "1" ]]; then
            echo "  ✓ skip [$m/$t] — S3 already at $s3"
            continue
        fi
        # Prereq: experiment 05 baseline must exist before staged LPAN
        baseline="results/multi_model/$m/$t/baseline/best_model"
        if [[ ! -f "$baseline/model.safetensors" ]]; then
            echo "  → [$m/$t] running baseline (experiment 05) first"
            wait_idle
            blog="logs/matrix/${m}_${t}_baseline.log"
            note_status "START baseline $m/$t"
            $PY experiments/05_multi_model_scaling.py --models "$m" --task "$t" \
                > "$blog" 2>&1 \
                && note_status "DONE  baseline $m/$t" \
                || { echo "    [FAIL baseline] $blog"; note_status "FAIL  baseline $m/$t"; continue; }
        fi
        echo "  → [$m/$t] launching staged LPAN (S1+S2+S3)"
        wait_idle
        log="logs/matrix/${m}_${t}_s1s2s3.log"
        note_status "START train $m/$t"
        $PY run_staged_lpan.py --model "$m" --task "$t" \
            > "$log" 2>&1 \
            && { echo "    [done] $log"; note_status "DONE  train $m/$t"; } \
            || { echo "    [FAIL] $log"; note_status "FAIL  train $m/$t"; }
    done
  done
fi

# ---------------------------------------------------------------------
# 2. Stage-4 range-aware for cells with S3 but no S4.
# ---------------------------------------------------------------------
if [[ " $STEPS " == *" s4 "* ]]; then
  echo "===== STEP 2: Stage-4 range-aware ====="
  for m in "${MODELS[@]}"; do
    for t in "${TASKS[@]}"; do
        s3="$(s3_dir "$m" "$t")"
        s4="$(s4_dir "$m" "$t")"
        if [[ ! -f "$s3/model.safetensors" ]]; then
            echo "  ⚠ skip [$m/$t] — S3 missing, train first"
            continue
        fi
        if [[ -f "$s4/model.safetensors" && "$FORCE" != "1" ]]; then
            echo "  ✓ skip [$m/$t] — S4 already at $s4"
            continue
        fi
        echo "  → [$m/$t] running Stage-4"
        wait_idle
        log="logs/matrix/${m}_${t}_s4.log"
        note_status "START s4    $m/$t"
        $PY run_stage4_range_aware.py --model "$m" --task "$t" \
            --no-keep-intervals --profile-on val --margin 0.30 \
            --epochs 3 --force --device cpu \
            > "$log" 2>&1 \
            && { echo "    [done] $log"; note_status "DONE  s4    $m/$t"; } \
            || { echo "    [FAIL] $log"; note_status "FAIL  s4    $m/$t"; }
    done
  done
fi

# ---------------------------------------------------------------------
# 3. Re-extract polynomial coefficients for cells with fresh S4.
# ---------------------------------------------------------------------
if [[ " $STEPS " == *" extract "* ]]; then
  echo "===== STEP 3: Coefficient extraction ====="
  for m in "${MODELS[@]}"; do
    for t in "${TASKS[@]}"; do
        s4="$(s4_dir "$m" "$t")"
        if [[ ! -f "$s4/model.safetensors" ]]; then
            echo "  ⚠ skip [$m/$t] — no S4 to extract from"
            continue
        fi
        echo "  → [$m/$t] extract coefficients"
        wait_idle
        log="logs/matrix/${m}_${t}_extract.log"
        note_status "START extract $m/$t"
        $PY extract_coefficients.py --model "$m" --task "$t" --source stage4 \
            > "$log" 2>&1 \
            && { echo "    [done] $log"; note_status "DONE  extract $m/$t"; } \
            || { echo "    [FAIL] $log"; note_status "FAIL  extract $m/$t"; }
    done
  done
fi

# ---------------------------------------------------------------------
# 4. SPO end-to-end validation per cell.
# ---------------------------------------------------------------------
if [[ " $STEPS " == *" validate "* ]]; then
  echo "===== STEP 4: SPO validation ====="
  for m in "${MODELS[@]}"; do
    for t in "${TASKS[@]}"; do
        s4="$(s4_dir "$m" "$t")"
        if [[ ! -f "$s4/model.safetensors" ]]; then
            echo "  ⚠ skip [$m/$t] — no S4 checkpoint"
            continue
        fi
        out="results/encrypted_inference/${m}_${t}_spo.json"
        if [[ -f "$out" && "$FORCE" != "1" ]]; then
            echo "  ✓ skip [$m/$t] — $out already exists"
            continue
        fi
        echo "  → [$m/$t] SPO validation (1 sample)"
        wait_idle
        log="logs/matrix/${m}_${t}_spo.log"
        note_status "START spo   $m/$t"
        $PY experiments/validate_e2e_spo.py \
            --model "$m" --task "$t" --num-samples 1 \
            --seq-len 4 --mult-depth 12 --ring-dim 16384 \
            --checkpoint "$s4" \
            > "$log" 2>&1 \
            && { echo "    [done] $log"; note_status "DONE  spo   $m/$t"; } \
            || { echo "    [FAIL] $log"; note_status "FAIL  spo   $m/$t"; }
    done
  done
fi

echo "===== matrix queue complete ====="
