# pybind11 wrapper for HEonGPU CKKS (Phase 1 smoke surface).
#
# This CMake project pulls HEonGPU in via add_subdirectory so we link the
# upstream `heongpu` static target directly, without needing a system
# install. Configure with:
#
#   cmake -S fhe_thesis/encryption/heongpu_bindings \
#         -B build_heongpu_py \
#         -G Ninja \
#         -D CMAKE_BUILD_TYPE=Release \
#         -D CMAKE_CUDA_ARCHITECTURES=90 \
#         -D HEONGPU_DIR=/workspace/HEonGPU \
#         -D Python3_EXECUTABLE=$(which python3)

cmake_minimum_required(VERSION 3.26)
project(heongpu_python LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED HEONGPU_DIR)
    message(FATAL_ERROR "HEONGPU_DIR must be set, e.g. -D HEONGPU_DIR=/workspace/HEonGPU")
endif()

# Pull in HEonGPU. We disable its examples/tests/benchmarks to keep the
# wrapper build fast — we only need the `heongpu` library target.
set(HEonGPU_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

add_subdirectory(${HEONGPU_DIR} ${CMAKE_BINARY_DIR}/_HEonGPU_build EXCLUDE_FROM_ALL)

# pybind11 — fetched via Python's installed package (pip install pybind11).
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    RESULT_VARIABLE PYBIND11_RESULT
)
if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not importable. Run: pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# bindings.cpp is intentionally compiled as CUDA so the device-link step
# performed by CMake for this target can resolve relocatable device code
# pulled in from the heongpu static library (which is built with
# CUDA_SEPARABLE_COMPILATION ON). Without this, the resulting .so misses
# the embedded fatbin and import fails with `undefined symbol: fatbinData`.
set_source_files_properties(bindings.cpp PROPERTIES LANGUAGE CUDA)

pybind11_add_module(_heongpu MODULE bindings.cpp)

set_target_properties(_heongpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_link_libraries(_heongpu PRIVATE
    heongpu
    CUDA::cudart
    CUDA::cudadevrt
    OpenMP::OpenMP_CXX
)

target_compile_options(_heongpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
