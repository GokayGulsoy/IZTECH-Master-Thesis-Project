#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GPUNTT::ntt" for configuration "Release"
set_property(TARGET GPUNTT::ntt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GPUNTT::ntt PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libntt-1.0.a"
  )

list(APPEND _cmake_import_check_targets GPUNTT::ntt )
list(APPEND _cmake_import_check_files_for_GPUNTT::ntt "${_IMPORT_PREFIX}/lib/libntt-1.0.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
