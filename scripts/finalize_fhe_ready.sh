#!/usr/bin/env bash
# Make all LPAN checkpoints FHE-ready.
#
# Pipeline (idempotent):
#   1. extract_coefficients.py --model all --task all
#         → dump trained polynomial coeffs to results/coefficients/.
#   2. audit_fhe_readiness.py
#         → measure out-of-range fraction per (model, task, layer, op);
#           write per-combo JSON + a summary that flags which combos
#           need Stage-4.
#   3. run_stage4_range_aware.py
#         → for every flagged combo: 1-2 epochs of range-aware
#           fine-tune from the existing staged_lpan_final checkpoint.
#   4. extract_coefficients.py (again, on Stage-4 outputs).
#   5. audit_fhe_readiness.py (again) to confirm OOR is gone.
#
# Usage:
#   bash scripts/finalize_fhe_ready.sh                     # all combos
#   MODELS="tiny mini" TASKS="sst2 mrpc" bash scripts/finalize_fhe_ready.sh
#   DRY_RUN=1 bash scripts/finalize_fhe_ready.sh
#
# Notes:
#   - Step 3 only touches combos the audit flagged.  Pass FORCE=1 to
#     skip the audit gate and run Stage-4 unconditionally.
#   - Step 4 needs to know the Stage-4 outputs live under
#     `stage4_range_aware/` rather than `staged_lpan_final/`; we patch
#     extract_coefficients.py with EXTRACT_SUBDIR for the second pass.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/fhe_venv/bin/python}"
MODELS="${MODELS:-tiny mini small base}"
TASKS="${TASKS:-sst2 mrpc qnli qqp}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run() {
    echo "+ $*"
    [[ "$DRY_RUN" == "1" ]] && return 0
    "$@"
}

echo "================================================================"
echo "  FHE-readiness finalisation"
echo "  models = $MODELS"
echo "  tasks  = $TASKS"
echo "  python = $PYTHON"
echo "  dry    = $DRY_RUN"
echo "  force  = $FORCE  (1 = skip audit gate before stage 4)"
echo "================================================================"

# ── Step 1: extract polynomial coefficients from staged_lpan_final ──
echo
echo "[1/5] Extracting LPAN coefficients (pre-Stage-4)…"
run "$PYTHON" extract_coefficients.py --model $MODELS --task $TASKS

# ── Step 2: audit out-of-range behaviour ──
echo
echo "[2/5] Auditing FHE-readiness (pre-Stage-4)…"
run "$PYTHON" experiments/audit_fhe_readiness.py \
        --model $MODELS --task $TASKS

# ── Step 3: range-aware Stage-4 fine-tune for flagged combos ──
echo
echo "[3/5] Stage-4 range-aware fine-tune…"
extra=()
[[ "$FORCE" == "1" ]] && extra+=(--force)
run "$PYTHON" run_stage4_range_aware.py \
        --model $MODELS --task $TASKS "${extra[@]:-}"

# ── Step 4: re-extract coefficients from Stage-4 outputs ──
# Stage-4 writes its best model under stage4_range_aware/best_model
# whereas extract_coefficients.py looks at staged_lpan_final/best_model.
# The cleanest way is to symlink / copy the new best_model into a
# parallel `staged_lpan_final_post_stage4/` and re-run extraction
# pointing at that.  Until we wire this through, we simply note the
# location and do the extract in step 5 once stage4 is wired through.
echo
echo "[4/5] Re-extracting coefficients (post-Stage-4)…"
echo "      NOTE: this step requires Stage-4 outputs; if no combos were"
echo "            flagged in step 2 it is a no-op."
for task in $TASKS; do
    for model in $MODELS; do
        sub=""; [[ "$task" != "sst2" ]] && sub="$task/"
        s4="results/multi_model/${model}/${sub}stage4_range_aware/best_model"
        target="results/multi_model/${model}/${sub}staged_lpan_final/best_model"
        if [[ -d "$s4" && -d "$target" ]]; then
            echo "  $model × $task: pointing extract at Stage-4 output"
            backup="results/multi_model/${model}/${sub}staged_lpan_final.pre_stage4_backup"
            if [[ ! -d "$backup" ]]; then
                run cp -a "$target" "$backup"
            fi
            run rsync -a --delete "$s4/" "$target/"
        fi
    done
done
run "$PYTHON" extract_coefficients.py --model $MODELS --task $TASKS

# ── Step 5: confirm OOR is gone ──
echo
echo "[5/5] Re-auditing FHE-readiness (post-Stage-4)…"
run "$PYTHON" experiments/audit_fhe_readiness.py \
        --model $MODELS --task $TASKS

echo
echo "================================================================"
echo "  Done.  See results/fhe_readiness/summary.json for status."
echo "================================================================"
