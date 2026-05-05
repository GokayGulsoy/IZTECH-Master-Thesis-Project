cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

message(VERBOSE "Executing download step for rapids-cmake")

block(SCOPE_FOR VARIABLES)

include("/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/CMakeFiles/fc-stamp/rapids-cmake/download-rapids-cmake.cmake")
include("/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/CMakeFiles/fc-stamp/rapids-cmake/verify-rapids-cmake.cmake")
include("/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/CMakeFiles/fc-stamp/rapids-cmake/extract-rapids-cmake.cmake")


endblock()
