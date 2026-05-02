#!/usr/bin/env bash
# ============================================================
#  Pod benchmark wrapper: pure LPAN vs HyPER-LPAN on MRPC.
#
#  Decides empirically whether HyPER-LPAN's primitive mixing
#  delivers a meaningful FHE speedup over pure LPAN under our
#  current packing / HEXL / pt×ct optimizations.
#
#  Run on the Pod after setup_pod.sh + checkpoint rsync:
#    bash /workspace/thesis/scripts/pod_run_mrpc_benchmarks.sh
#
#  Time: ~2× (n_samples × per-sample) ≈ 30-90 min for 20 samples.
#  Memory peak: ~110 GB (rotation keys at ring_dim=65536).
# ============================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/thesis}"
VENV="$REPO_ROOT/venv"
N_SAMPLES="${N_SAMPLES:-20}"
N_JOBS="${N_JOBS:-$(nproc)}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/fhe_benchmarks/mrpc_$(date +%Y%m%d_%H%M)}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "  MRPC FHE benchmark (pure LPAN vs HyPER-LPAN v4b)"
echo "  n_samples   = $N_SAMPLES"
echo "  n_jobs      = $N_JOBS"
echo "  max_seq_len = $MAX_SEQ_LEN"
echo "  out_dir     = $OUT_DIR"
echo "=============================================="

# ── Verify checkpoints present ───────────────────────────────────────
# v3 = highest-accuracy HyPER-LPAN run (F1=0.8554), budget 0.67.
# Layer plan: Quad@[0,1,2,3,4,7,8,9,10,11], LPAN@[5,6].
LPAN_CKPT="results/multi_model/mrpc/base/staged_lpan_final/best_model"
HYPER_CKPT="results/multi_model/mrpc/base/hyper_lpan_v1_85.54f1/best_model"
for ckpt in "$LPAN_CKPT" "$HYPER_CKPT"; do
    if [[ ! -d "$ckpt" ]]; then
        echo "ERROR: missing checkpoint: $ckpt"
        echo "       Run scripts/pod_rsync_checkpoints.sh from local first."
        exit 1
    fi
done
echo "  checkpoints OK ✓"

# ── Bench 1: pure LPAN ───────────────────────────────────────────────
echo
echo "[1/2] Pure LPAN baseline (b=1.00, all 12 layers LPAN)..."
echo "      Expected per-sample: 12-25s (depends on packing efficiency)"
LPAN_LOG="$OUT_DIR/lpan_baseline.log"
python experiments/run_fhe_benchmark.py \
    --model base --task mrpc \
    --n-samples "$N_SAMPLES" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --n-jobs "$N_JOBS" \
    --checkpoint "$LPAN_CKPT" \
    --out "$OUT_DIR/lpan" \
    2>&1 | tee "$LPAN_LOG"

# ── Bench 2: HyPER-LPAN v4b (LPAN×8 + Quad×4 from MCKP plan) ──────
echo
echo "[2/2] HyPER-LPAN v3 (b=0.67: Quad@[0,1,2,3,4,7,8,9,10,11], LPAN@[5,6])..."
echo "      Expected per-sample: 6-14s"
HYPER_LOG="$OUT_DIR/hyper_lpan.log"
python experiments/run_fhe_benchmark.py \
    --model base --task mrpc \
    --n-samples "$N_SAMPLES" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --n-jobs "$N_JOBS" \
    --hybrid \
    --linear-mixing-layers "" \
    --quad-attention-layers "0,1,2,3,4,7,8,9,10,11" \
    --checkpoint "$HYPER_CKPT" \
    --out "$OUT_DIR/hyper" \
    2>&1 | tee "$HYPER_LOG"

# ── Summary ──────────────────────────────────────────────────────────
echo
echo "=============================================="
echo "  Results summary"
echo "=============================================="
python - <<PYEOF
import json, glob, os
from pathlib import Path
out = Path("$OUT_DIR")
for label, sub in [("LPAN baseline (b=1.00, F1=0.8795)", "lpan"),
                   ("HyPER-LPAN v3   (b=0.67, F1=0.8554)", "hyper")]:
    files = list((out / sub).glob("fhe_benchmark_*.json"))
    if not files:
        print(f"  {label}: NO RESULTS")
        continue
    d = json.loads(files[0].read_text())
    print(f"\n  {label}:")
    print(f"    plain acc/F1   = {d.get('plaintext_accuracy', 0)*100:.2f}%")
    print(f"    fhe acc/F1     = {d.get('fhe_accuracy', 0)*100:.2f}%")
    print(f"    mean latency   = {d.get('mean_latency_s', 0):.2f}s/sample")
    print(f"    median latency = {d.get('median_latency_s', 0):.2f}s/sample")
    print(f"    min / max      = {d.get('min_latency_s', 0):.2f}s / {d.get('max_latency_s', 0):.2f}s")
PYEOF

echo
echo "Logs: $OUT_DIR/{lpan,hyper}.log"
echo "JSONs: $OUT_DIR/{lpan,hyper}/fhe_benchmark_*.json"
