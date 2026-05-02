#!/usr/bin/env bash
# ============================================================
#  Rsync only the files needed for FHE benchmarks to the Pod.
#
#  Run from LOCAL machine (this repo root) AFTER pod_wipe_and_clone.sh:
#    POD_HOST=runpod-xxx.runpod.io POD_PORT=22001 \
#      bash scripts/pod_rsync_checkpoints.sh
#
#  Total transfer: ~450 MB (BERT-base × 2 checkpoints + cache)
#  Time on 100 Mbps: ~1 min
# ============================================================
set -euo pipefail

POD_HOST="${POD_HOST:?set POD_HOST=user@host or runpod hostname}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/thesis}"

# Compose ssh+rsync flags
SSH_OPTS="-p $POD_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
RSYNC_FLAGS="-avzP --human-readable -e \"ssh $SSH_OPTS\""

echo "=============================================="
echo "  Rsyncing checkpoints to Pod"
echo "  pod = $POD_USER@$POD_HOST:$POD_PORT"
echo "  dst = $REMOTE_REPO"
echo "=============================================="

# Helper for repeated rsync invocations
push() {
    local src="$1" dst="$2"
    if [[ ! -e "$src" ]]; then
        echo "  [skip] $src (does not exist)"
        return
    fi
    echo
    echo "  → $src"
    eval rsync $RSYNC_FLAGS "\"$src\"" \
        "$POD_USER@$POD_HOST:\"$REMOTE_REPO/$dst\""
}

# ── 1. LPAN baseline checkpoint (Stage A reference) ─────────────────
push "results/multi_model/mrpc/base/staged_lpan_final/" \
     "results/multi_model/mrpc/base/"

# ── 2. HyPER-LPAN v3 checkpoint (highest-accuracy run, F1=0.8554) ───
#  v3 = budget 0.67, Quad@[0,1,2,3,4,7,8,9,10,11], LPAN@[5,6]
push "results/multi_model/mrpc/base/hyper_lpan_v1_85.54f1/" \
     "results/multi_model/mrpc/base/"

# ── 3. FP32 baseline cache (if populated) ───────────────────────────
push "results/baselines/fp32_finetuned.json" \
     "results/baselines/fp32_finetuned.json"

# ── 4. Polynomial coefficients (small but required) ─────────────────
push "results/multi_model/coefficients/" \
     "results/multi_model/"

# ── 5. MCKP plan (selects which layers are quad/lpan/lm) ────────────
push "results/composition/" \
     "results/"

echo
echo "=============================================="
echo "  Rsync complete."
echo "  Now ssh to Pod and run:"
echo "    bash $REMOTE_REPO/scripts/setup_pod.sh"
echo "    bash $REMOTE_REPO/scripts/pod_run_mrpc_benchmarks.sh"
echo "=============================================="
