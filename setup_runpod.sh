#!/bin/bash
# ================================================================
# setup_runpod.sh  –  Bootstrap LPAN FHE Thesis on RunPod A100
# ================================================================
# Usage (run once after SSH-ing into the pod):
#   chmod +x setup_runpod.sh && bash setup_runpod.sh
# ================================================================
set -e

REPO_URL="https://github.com/GokayGulsoy/IZTECH-Master-Thesis-Project.git"
BRANCH="main"
PROJECT_DIR="/workspace/thesis"
VENV_DIR="$PROJECT_DIR/venv"

echo "================================================================"
echo " LPAN FHE RunPod Bootstrap"
echo " Target: $PROJECT_DIR"
echo "================================================================"

# ── 1. System packages ───────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -q
apt-get install -y -q git curl wget build-essential python3-dev python3-venv python3-pip

# ── 2. Detect CUDA version ───────────────────────────────────────
echo "[2/7] Detecting CUDA version..."
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "  nvcc reported: $CUDA_VERSION"
elif [ -f /usr/local/cuda/version.json ]; then
    CUDA_VERSION=$(python3 -c "import json; d=json.load(open('/usr/local/cuda/version.json')); print(d['cuda']['version'])")
    echo "  version.json reported: $CUDA_VERSION"
else
    CUDA_VERSION="12.1"
    echo "  WARNING: Could not detect CUDA version, defaulting to $CUDA_VERSION"
fi

# Map full version to torch wheel suffix
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"

# Resolve PyTorch version compatible with detected CUDA
if   [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]; then
    TORCH_VERSION="2.7.0+cu128"
    TORCH_CUDA_TAG="cu128"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_VERSION="2.4.1+cu124"
    TORCH_CUDA_TAG="cu124"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 1 ]]; then
    TORCH_VERSION="2.3.1+cu121"
    TORCH_CUDA_TAG="cu121"
else
    TORCH_VERSION="2.3.1+cu118"
    TORCH_CUDA_TAG="cu118"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
echo "  CUDA detected: $CUDA_VERSION  →  Installing torch $TORCH_VERSION"

# ── 3. Clone repo ────────────────────────────────────────────────
echo "[3/7] Cloning repository ($BRANCH branch)..."
if [ -d "$PROJECT_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# ── 4. Create Python venv ────────────────────────────────────────
echo "[4/7] Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 5. Install PyTorch ───────────────────────────────────────────
echo "[5/7] Installing PyTorch ${TORCH_VERSION} for CUDA ${CUDA_VERSION}..."
pip install \
    torch==${TORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url "$TORCH_INDEX" \
    --quiet

# ── 6. Install project requirements ─────────────────────────────
echo "[6/7] Installing project requirements..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

# ── 7. Verification gates ─────────────────────────────────────────
echo "[7/7] Running verification gates..."
python3 - <<'PYEOF'
import sys

# Gate 1: PyTorch + CUDA
import torch
cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
print(f"  [{'OK' if cuda_ok else 'FAIL'}] torch {torch.__version__}  CUDA={cuda_ok}  GPU={gpu_name}")
if not cuda_ok:
    print("  WARNING: CUDA not available in PyTorch. Check CUDA installation.")

# Gate 2: OpenFHE
try:
    import openfhe
    params = openfhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(2)
    params.SetScalingModSize(50)
    cc = openfhe.GenCryptoContext(params)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    keys = cc.KeyGen()
    print(f"  [OK] openfhe {openfhe.__version__}  CKKS context created")
except Exception as e:
    print(f"  [FAIL] openfhe import/context: {e}")

# Gate 3: Transformers
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    out = tok("hello world", return_tensors="pt")
    print(f"  [OK] transformers loaded BERT-Tiny tokenizer  ids={list(out['input_ids'].shape)}")
except Exception as e:
    print(f"  [FAIL] transformers: {e}")

# Gate 4: GPU memory (if CUDA available)
if cuda_ok:
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  [OK] GPU VRAM: {total_gb:.1f} GB total")
    if total_gb < 30:
        print(f"  WARNING: VRAM {total_gb:.1f} GB may be insufficient for BERT-Base encrypted inference")

print("\n  Setup complete. Activate venv with:")
print(f"    source {sys.argv[0].replace('setup_runpod.sh','venv/bin/activate') if __name__ != '__main__' else '$PROJECT_DIR/venv/bin/activate'}")
PYEOF

echo "================================================================"
echo " Bootstrap finished."
echo " Activate the venv:"
echo "   source $VENV_DIR/bin/activate"
echo " Run LPAN training:"
echo "   python run_staged_lpan.py --model_name prajjwal1/bert-tiny --task sst2 --output_dir results/bert-tiny-sst2"
echo "================================================================"
