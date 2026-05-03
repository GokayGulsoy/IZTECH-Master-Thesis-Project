#!/usr/bin/env bash
# Build the HEonGPU pybind11 wrapper. Defaults match the H100 Pod layout.
set -euo pipefail

HEONGPU_DIR=${HEONGPU_DIR:-/workspace/HEonGPU}
CUDA_ARCH=${CUDA_ARCH:-90}
BUILD_DIR=${BUILD_DIR:-build_heongpu_py}
JOBS=${JOBS:-$(nproc)}

export PATH=/usr/local/cuda/bin:${PATH}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m pip install --quiet --upgrade "pybind11>=2.12"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -D HEONGPU_DIR="${HEONGPU_DIR}" \
    -D Python3_EXECUTABLE="$(which python3)"

cmake --build "${BUILD_DIR}" -j"${JOBS}"

# Surface the freshly built .so next to the Python wrapper for easy import.
SO_PATH=$(find "${BUILD_DIR}" -maxdepth 2 -name "_heongpu*.so" | head -1)
if [[ -z "${SO_PATH}" ]]; then
    echo "ERROR: built _heongpu*.so not found under ${BUILD_DIR}" >&2
    exit 1
fi
cp -f "${SO_PATH}" "${SCRIPT_DIR}/"
echo "Installed: ${SCRIPT_DIR}/$(basename "${SO_PATH}")"
