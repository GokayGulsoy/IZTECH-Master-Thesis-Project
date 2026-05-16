#!/usr/bin/env bash
set -euo pipefail

if ! (return 0 2>/dev/null); then
    echo "Use: source scripts/activate_pod_env.sh" >&2
    exit 1
fi

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/repo}"
VENV_DIR="${VENV_DIR:-$WORKSPACE/fhe_venv}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "Missing repo directory: $REPO_DIR" >&2
    return 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Missing pod venv: $VENV_DIR" >&2
    echo "Run: bash $REPO_DIR/scripts/setup_pod_gpu.sh" >&2
    return 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

cd "$REPO_DIR"
echo "Pod environment ready: $REPO_DIR"
echo "Python: $(command -v python)"