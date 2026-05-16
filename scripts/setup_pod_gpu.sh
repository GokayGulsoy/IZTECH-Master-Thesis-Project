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
VENV_DIR="${VENV_DIR:-$WORKSPACE/fhe_venv}"

REPO_URL="${REPO_URL:-https://github.com/GokayGulsoy/IZTECH-Master-Thesis-Project.git}"
REPO_BRANCH="${REPO_BRANCH:-synthesizer-lpan-production}"

HEONGPU_URL="${HEONGPU_URL:-https://github.com/Alisah-Ozcan/HEonGPU.git}"
HEONGPU_COMMIT="${HEONGPU_COMMIT:-f91381a1b73da33118b0a1d511b4e81bf943ed83}"
SYNC_REPOS="${SYNC_REPOS:-1}"

CUDA_ARCH="${CUDA_ARCH:-90}"   # H100 = sm_90
NPROC=$(nproc)

export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

echo "=============================================="
echo "  Synthesizer-LPAN GPU pod setup"
echo "  workspace = $WORKSPACE"
echo "  repo      = $REPO_DIR ($REPO_BRANCH)"
echo "  HEonGPU   = $HEONGPU_DIR @ $HEONGPU_COMMIT"
echo "  venv      = $VENV_DIR"
echo "  sync git  = $SYNC_REPOS"
echo "  CUDA arch = sm_$CUDA_ARCH"
echo "  cores     = $NPROC"
echo "=============================================="

# ── 0. Verify GPU ──────────────────────────────────────────────────────
echo
echo "[0/6] GPU check..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | tail -1

# ── 1. Apt deps ────────────────────────────────────────────────────────
echo
echo "[1/6] System packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ninja-build g++ gcc python3-dev python3-pip \
    git ca-certificates wget curl build-essential \
    libssl-dev libgmp-dev libomp-dev libntl-dev

# ── 2. Repos ───────────────────────────────────────────────────────────
echo
echo "[2/6] Cloning repos..."
mkdir -p "$WORKSPACE"
if [[ "$SYNC_REPOS" == "1" ]]; then
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        git clone "$REPO_URL" "$REPO_DIR"
        git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    else
        git -C "$REPO_DIR" fetch --all -q
        git -C "$REPO_DIR" checkout "$REPO_BRANCH"
        git -C "$REPO_DIR" pull -q
    fi
elif [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "ERROR: SYNC_REPOS=0 requires an existing git repo at $REPO_DIR" >&2
    exit 1
fi
echo "  repo HEAD: $(git -C $REPO_DIR rev-parse --short HEAD)"

if [[ "$SYNC_REPOS" == "1" ]]; then
    if [[ ! -d "$HEONGPU_DIR/.git" ]]; then
        git clone "$HEONGPU_URL" "$HEONGPU_DIR"
    fi
    git -C "$HEONGPU_DIR" fetch --all -q
    git -C "$HEONGPU_DIR" checkout "$HEONGPU_COMMIT"
elif [[ ! -d "$HEONGPU_DIR/.git" ]]; then
    echo "ERROR: SYNC_REPOS=0 requires an existing git repo at $HEONGPU_DIR" >&2
    exit 1
fi
echo "  HEonGPU HEAD: $(git -C $HEONGPU_DIR rev-parse --short HEAD)"

# ── 3. Persistent Python env/tooling ────────────────────────────────────
echo
echo "[3/6] Persistent Python env..."
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Apt's cmake (3.22) is too old for HEonGPU (>=3.26.4). Keep newer build
# tooling inside the persistent venv so same-pod resumes do not need a
# global pip reinstall.
python -m pip install --upgrade pip setuptools wheel -q
python -m pip install --quiet --upgrade "cmake>=3.28" "pybind11>=2.12"
export PATH="$VENV_DIR/bin:/usr/local/cuda/bin:$PATH"
cmake --version | head -1
ninja --version
python --version

# ── 4. Build HEonGPU ────────────────────────────────────────────────────
echo
echo "[4/6] Building HEonGPU (cached under /workspace)..."
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
    echo "  HEonGPU build cache ready ✓"

    # ── 5. Python deps ─────────────────────────────────────────────────────
echo
    echo "[5/6] Python deps..."
# Install a CUDA-enabled torch build first, then let pyproject.toml pull the
# remaining project dependencies.
    python -m pip install --upgrade torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" -q
    python -m pip install -e "$REPO_DIR" -q
echo "  pip deps installed ✓"

    # ── 6. Build bindings ──────────────────────────────────────────────────
echo
    echo "[6/6] Building HEonGPU pybind11 bindings..."
cd "$REPO_DIR/fhe_thesis/encryption/heongpu_bindings"
HEONGPU_DIR="$HEONGPU_DIR" CUDA_ARCH="$CUDA_ARCH" bash build.sh
echo
echo "=============================================="
    echo "  Setup complete. Same-pod resume command:"
    echo "    source $REPO_DIR/scripts/activate_pod_env.sh"
    echo ""
    echo "  Quick smoke test:"
    echo "    cd $REPO_DIR && python scripts/smoke_heongpu_backend.py"
echo "=============================================="
