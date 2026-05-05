#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "RNGonGPU::rngongpu" for configuration "Release"
set_property(TARGET RNGonGPU::rngongpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(RNGonGPU::rngongpu PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librngongpu-1.0.a"
  )

list(APPEND _cmake_import_check_targets RNGonGPU::rngongpu )
list(APPEND _cmake_import_check_files_for_RNGonGPU::rngongpu "${_IMPORT_PREFIX}/lib/librngongpu-1.0.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
