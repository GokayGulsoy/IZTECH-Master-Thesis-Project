#!/usr/bin/env bash
# ============================================================
#  Pod cleanup + fresh clone of HyPER-LPAN repo.
#
#  Run on the Pod after SSH-ing in (BEFORE running setup_pod.sh):
#    bash <(curl -fsSL <raw-url>/scripts/pod_wipe_and_clone.sh)
#  Or after rsync of this script:
#    bash /workspace/pod_wipe_and_clone.sh
#
#  After this completes:
#    bash /workspace/thesis/scripts/setup_pod.sh
# ============================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/thesis"
REPO_URL="${REPO_URL:-https://github.com/GokayGulsoy/IZTECH-Master-Thesis-Project.git}"
REPO_BRANCH="${REPO_BRANCH:-feature/hyper-lpan-extensions}"

echo "=============================================="
echo "  Pod wipe & re-clone"
echo "  workspace = $WORKSPACE"
echo "  repo dir  = $REPO_DIR"
echo "  repo url  = $REPO_URL"
echo "  branch    = $REPO_BRANCH"
echo "=============================================="

# ── 1. Stop any running benchmarks ────────────────────────────────────
echo
echo "[1/4] Killing any in-flight FHE/training jobs..."
pkill -f "run_fhe_benchmark" 2>/dev/null || true
pkill -f "train_hyper_lpan" 2>/dev/null || true
sleep 1
echo "  ok"

# ── 2. Wipe old repo + venv + tmp build artifacts ─────────────────────
echo
echo "[2/4] Wiping previous state..."
if [[ -d "$REPO_DIR" ]]; then
    echo "  removing $REPO_DIR"
    rm -rf "$REPO_DIR"
fi
# Keep /tmp/fhe_build (HEXL + OpenFHE) — those are slow to rebuild and
# version-stable. Re-running setup_pod.sh will skip rebuild if already present.
echo "  preserving /tmp/fhe_build (HEXL + OpenFHE C++ libs)"
echo "  ok"

# ── 3. Fresh clone ────────────────────────────────────────────────────
echo
echo "[3/4] Cloning fresh repo..."
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" thesis
cd "$REPO_DIR"
echo "  HEAD: $(git log -1 --oneline)"
echo "  ok"

# ── 4. Prepare directories for checkpoint rsync ───────────────────────
echo
echo "[4/4] Creating checkpoint dirs (to receive rsync from local)..."
mkdir -p \
    "$REPO_DIR/results/multi_model/mrpc/base/staged_lpan_final/best_model" \
    "$REPO_DIR/results/multi_model/mrpc/base/hyper_lpan_v1_85.54f1/best_model" \
    "$REPO_DIR/results/baselines" \
    "$REPO_DIR/logs"
# v3 checkpoint (F1=0.8554, our highest-accuracy HyPER-LPAN run) is what
# pod_run_mrpc_benchmarks.sh evaluates against pure LPAN.
echo "  ok"

echo
echo "=============================================="
echo "  Wipe & clone complete."
echo
echo "  Next steps (from your LOCAL machine):"
echo "    1. rsync checkpoints (use scripts/pod_rsync_checkpoints.sh)"
echo "    2. ssh back to pod and run:"
echo "       bash $REPO_DIR/scripts/setup_pod.sh"
echo "=============================================="
