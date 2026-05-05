#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "HEonGPU::heongpu" for configuration "Release"
set_property(TARGET HEonGPU::heongpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HEonGPU::heongpu PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libheongpu.a"
  )

list(APPEND _cmake_import_check_targets HEonGPU::heongpu )
list(APPEND _cmake_import_check_files_for_HEonGPU::heongpu "/usr/local/lib/libheongpu.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
