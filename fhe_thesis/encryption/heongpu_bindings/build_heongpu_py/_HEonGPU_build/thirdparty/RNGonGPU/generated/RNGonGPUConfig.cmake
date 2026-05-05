include(CMakeFindDependencyMacro)

find_dependency(CUDAToolkit REQUIRED)
find_dependency(OpenSSL REQUIRED)
find_dependency(GPUNTT REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/RNGonGPUTargets.cmake")
