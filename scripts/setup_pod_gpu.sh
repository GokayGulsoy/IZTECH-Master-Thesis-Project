#!/usr/bin/env bash
# ============================================================
#  GPU Pod setup: HEonGPU + Python deps + bindings
#  Target: RunPod H100 (CUDA 12.4, Ubuntu 22.04, Python 3.11)
#
#  Run once after pod creation OR after a wipe/migration:
#    bash setup_pod_gpu.sh 2>&1 | tee /workspace/setup_gpu.log
#
#  Idempotent: re-runs skip already-completed steps.
# ============================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/repo}"
HEONGPU_DIR="${HEONGPU_DIR:-$WORKSPACE/HEonGPU}"

REPO_URL="${REPO_URL:-https://github.com/GokayGulsoy/IZTECH-Master-Thesis-Project.git}"
REPO_BRANCH="${REPO_BRANCH:-feature/matrix-packing-and-gpu}"

HEONGPU_URL="${HEONGPU_URL:-https://github.com/Alisah-Ozcan/HEonGPU.git}"
HEONGPU_COMMIT="${HEONGPU_COMMIT:-f91381a1b73da33118b0a1d511b4e81bf943ed83}"

CUDA_ARCH="${CUDA_ARCH:-90}"   # H100 = sm_90
NPROC=$(nproc)

export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

echo "=============================================="
echo "  LPAN-FHE GPU pod setup"
echo "  workspace = $WORKSPACE"
echo "  repo      = $REPO_DIR ($REPO_BRANCH)"
echo "  HEonGPU   = $HEONGPU_DIR @ $HEONGPU_COMMIT"
echo "  CUDA arch = sm_$CUDA_ARCH"
echo "  cores     = $NPROC"
echo "=============================================="

# ── 0. Verify GPU ──────────────────────────────────────────────────────
echo
echo "[0/5] GPU check..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | tail -1

# ── 1. Apt deps ────────────────────────────────────────────────────────
echo
echo "[1/5] System packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cmake ninja-build g++ gcc python3-dev python3-pip \
    git ca-certificates wget curl build-essential

cmake --version | head -1
ninja --version

# ── 2. Repos ───────────────────────────────────────────────────────────
echo
echo "[2/5] Cloning repos..."
mkdir -p "$WORKSPACE"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
else
    git -C "$REPO_DIR" fetch --all -q
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull -q
fi
echo "  repo HEAD: $(git -C $REPO_DIR rev-parse --short HEAD)"

if [[ ! -d "$HEONGPU_DIR/.git" ]]; then
    git clone "$HEONGPU_URL" "$HEONGPU_DIR"
fi
git -C "$HEONGPU_DIR" fetch --all -q
git -C "$HEONGPU_DIR" checkout "$HEONGPU_COMMIT"
echo "  HEonGPU HEAD: $(git -C $HEONGPU_DIR rev-parse --short HEAD)"

# ── 3. Build & install HEonGPU ─────────────────────────────────────────
echo
echo "[3/5] Building HEonGPU (this is the slow step ~15-30 min)..."
HEO_BUILD="$HEONGPU_DIR/build"
if [[ ! -f "$HEO_BUILD/CMakeCache.txt" ]]; then
    cmake -S "$HEONGPU_DIR" -B "$HEO_BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DHEonGPU_BUILD_TESTS=OFF \
        -DHEonGPU_BUILD_EXAMPLES=OFF \
        -DHEonGPU_BUILD_BENCHMARKS=OFF
fi
cmake --build "$HEO_BUILD" -j"$NPROC"
cmake --install "$HEO_BUILD" --prefix /usr/local
ldconfig
echo "  HEonGPU installed ✓"

# ── 4. Python deps ─────────────────────────────────────────────────────
echo
echo "[4/5] Python deps..."
python3 -m pip install --upgrade pip setuptools wheel -q
python3 -m pip install --upgrade "pybind11>=2.12" -q
# Repo-level deps (skip torch — already preinstalled in container)
grep -vE "^(torch|torchvision|torchaudio|openfhe)" "$REPO_DIR/requirements.txt" \
    > /tmp/req_no_torch.txt || true
python3 -m pip install -r /tmp/req_no_torch.txt -q
echo "  pip deps installed ✓"

# ── 5. Build bindings ──────────────────────────────────────────────────
echo
echo "[5/5] Building HEonGPU pybind11 bindings..."
cd "$REPO_DIR/fhe_thesis/encryption/heongpu_bindings"
HEONGPU_DIR="$HEONGPU_DIR" CUDA_ARCH="$CUDA_ARCH" bash build.sh
echo
echo "=============================================="
echo "  Setup complete. Quick smoke test:"
echo "    cd $REPO_DIR && PYTHONPATH=. python -c \\"
echo "      'from fhe_thesis.encryption import heongpu_bindings as h; print(h.CudaStream)'"
echo "=============================================="
