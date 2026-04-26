#!/usr/bin/env bash
# ============================================================
#  Pod setup: OpenFHE + HEXL (AVX-512) + Python env
#  Target: RunPod 32-vCPU / 256 GB Memory-Optimized CPU pod
#          (Intel Ice Lake / Sapphire Rapids, AVX-512)
#
#  Run once after pod creation:
#    bash /workspace/thesis/scripts/setup_pod.sh 2>&1 | tee /workspace/setup.log
#
#  After completion, activate and benchmark:
#    source /workspace/thesis/venv/bin/activate
#    python scripts/openfhe_smoke_bootstrap.py   # < 5 min
#    python experiments/run_fhe_benchmark.py --model base --task sst2 \
#        --n-samples 100 --max-seq-len 64 --n-jobs 32 \
#        --checkpoint results/multi_model/sst2/base/staged_lpan_final/best_model
# ============================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/thesis}"
BUILD_DIR="${BUILD_DIR:-/tmp/fhe_build}"
VENV="$REPO_ROOT/venv"
NPROC=$(nproc)

echo "=============================================="
echo "  LPAN-FHE pod setup"
echo "  repo      = $REPO_ROOT"
echo "  build dir = $BUILD_DIR"
echo "  cores     = $NPROC"
echo "=============================================="

# ── 0. Verify AVX-512 (required for HEXL) ─────────────────────────────
echo
echo "[0/6] Checking AVX-512 support..."
if ! grep -q avx512 /proc/cpuinfo; then
    echo "ERROR: This CPU does not support AVX-512."
    echo "       Select an Intel Ice Lake or Sapphire Rapids pod on RunPod."
    exit 1
fi
echo "  AVX-512 detected ✓"
AVX512_FLAGS=$(grep -o 'avx512[a-z_]*' /proc/cpuinfo | sort -u | tr '\n' ' ')
echo "  Flags: $AVX512_FLAGS"

# ── 1. System dependencies ─────────────────────────────────────────────
echo
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    cmake g++ gcc python3-dev python3-pip python3-venv \
    libgmp-dev libomp-dev git autoconf automake libtool \
    wget curl ca-certificates

# ── 2. Intel HEXL (AVX-512 NTT accelerator) ───────────────────────────
echo
echo "[2/6] Building Intel HEXL..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

if [[ ! -d hexl ]]; then
    git clone --depth 1 https://github.com/intel/hexl.git
fi
cd hexl
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DHEXL_TESTING=OFF \
    -DHEXL_BENCHMARK=OFF \
    -DHEXL_COVERAGE=OFF
cmake --build build -j"$NPROC"
cmake --install build --prefix /usr/local
echo "  HEXL installed ✓"
cd "$BUILD_DIR"

# ── 3. OpenFHE C++ library (with HEXL + OpenMP) ───────────────────────
echo
echo "[3/6] Building OpenFHE (with HEXL + OpenMP)..."
# Pin to a known-good release tag for reproducibility.
OPENFHE_TAG="${OPENFHE_TAG:-v1.2.3}"

if [[ ! -d openfhe-development ]]; then
    git clone --depth 1 --branch "$OPENFHE_TAG" \
        https://github.com/openfheorg/openfhe-development.git
fi
cd openfhe-development
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_HEXL=ON \
    -DWITH_OPENMP=ON \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local
make -j"$NPROC"
make install
ldconfig  # update shared library cache
echo "  OpenFHE $OPENFHE_TAG installed ✓"
cd "$BUILD_DIR"

# ── 4. openfhe-python bindings ─────────────────────────────────────────
echo
echo "[4/6] Building openfhe-python bindings..."

# Ensure the venv exists before building.
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel pybind11 -q

# v0.8.10 is the last release that targets OpenFHE 1.2.x.
# Later tags (v1.3+) require OpenFHE ≥1.3 and are not compatible.
OPENFHE_PY_TAG="v0.8.10"
if [[ ! -d "$BUILD_DIR/openfhe-python" ]]; then
    git clone --depth 1 --branch "$OPENFHE_PY_TAG" \
        https://github.com/openfheorg/openfhe-python.git \
        "$BUILD_DIR/openfhe-python"
fi
cd "$BUILD_DIR/openfhe-python"

# Always use direct CMake build (pip install does not forward OpenFHE_DIR).
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(python3 -c 'import sys; print(sys.executable)')" \
    -DOpenFHE_DIR=/usr/local/lib/OpenFHE \
    -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
make -j"$NPROC"

# Install .so directly into venv site-packages.
SOFILE=$(find . -name "openfhe*.so" | head -1)
if [[ -z "$SOFILE" ]]; then
    echo "ERROR: openfhe .so not found after build"
    exit 1
fi
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp "$SOFILE" "$SITE/"
echo "  Installed $SOFILE → $SITE/"
echo "  openfhe-python $OPENFHE_PY_TAG installed ✓"
cd "$BUILD_DIR"

# ── 5. Python dependencies ─────────────────────────────────────────────
echo
echo "[5/6] Installing Python dependencies..."
source "$VENV/bin/activate"

pip install --upgrade pip

# Core ML stack (CPU-only torch — no CUDA needed for FHE)
pip install \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.41.2 \
    datasets==2.20.0 \
    scikit-learn \
    numpy \
    tqdm

# Install the thesis package in editable mode
cd "$REPO_ROOT"
pip install -e . 2>/dev/null || pip install -e . --no-build-isolation 2>/dev/null || true

echo "  Python deps installed ✓"

# ── 6. Smoke tests ─────────────────────────────────────────────────────
echo
echo "[6/6] Running smoke tests..."
source "$VENV/bin/activate"
cd "$REPO_ROOT"

# Quick backend smoke test (small ring_dim, no bootstrap)
python - <<'PYEOF'
import openfhe as ofhe
print(f"  OpenFHE version: {ofhe.__version__ if hasattr(ofhe, '__version__') else 'unknown'}")
params = ofhe.CCParamsCKKSRNS()
params.SetSecretKeyDist(ofhe.SecretKeyDist.UNIFORM_TERNARY)
params.SetSecurityLevel(ofhe.SecurityLevel.HEStd_NotSet)
params.SetKeySwitchTechnique(ofhe.KeySwitchTechnique.HYBRID)
params.SetRingDim(1 << 12)
params.SetMultiplicativeDepth(5)
params.SetBatchSize(8)
cc = ofhe.GenCryptoContext(params)
cc.Enable(ofhe.PKESchemeFeature.PKE)
cc.Enable(ofhe.PKESchemeFeature.KEYSWITCH)
cc.Enable(ofhe.PKESchemeFeature.LEVELEDSHE)
keys = cc.KeyGen()
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
pt = cc.MakeCKKSPackedPlaintext(x, 1, 0, None, 8)
ct = cc.Encrypt(keys.publicKey, pt)
cc.EvalMultKeyGen(keys.secretKey)
ct2 = cc.EvalMult(ct, ct)
pt2 = cc.Decrypt(ct2, keys.secretKey)
pt2.SetLength(8)
vals = list(pt2.GetRealPackedValue())
ref = [v**2 for v in x]
err = max(abs(a-b) for a,b in zip(vals, ref))
assert err < 1e-3, f"Smoke test failed: err={err}"
print(f"  CKKS multiply smoke test PASSED (max_err={err:.2e})")
print(f"  HYBRID key switching: OK")
PYEOF

# AVX-512 NTT in action: bootstrap smoke
echo "  Running bootstrap smoke test (N=4096, ~30-60s)..."
python scripts/openfhe_smoke_bootstrap.py
echo "  Bootstrap smoke test: PASSED ✓"

echo
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Activate env:  source $VENV/bin/activate"
echo ""
echo "  Quick benchmark (100 SST-2 samples):"
echo "    python experiments/run_fhe_benchmark.py \\"
echo "        --model base --task sst2 \\"
echo "        --n-samples 100 --max-seq-len 64 --n-jobs $NPROC \\"
echo "        --checkpoint results/multi_model/sst2/base/staged_lpan_final/best_model"
echo ""
echo "  Single-sentence test:"
echo "    python scripts/run_inference.py \\"
echo "        --task sst2 --model base --n-jobs $NPROC \\"
echo "        --text 'This movie was absolutely fantastic!'"
echo "=============================================="
