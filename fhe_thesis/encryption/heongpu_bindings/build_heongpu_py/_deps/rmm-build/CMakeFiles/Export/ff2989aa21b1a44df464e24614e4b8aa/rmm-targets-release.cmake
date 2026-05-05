#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rmm::rmm" for configuration "Release"
set_property(TARGET rmm::rmm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rmm::rmm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librmm.so"
  IMPORTED_SONAME_RELEASE "librmm.so"
  )

list(APPEND _cmake_import_check_targets rmm::rmm )
list(APPEND _cmake_import_check_files_for_rmm::rmm "${_IMPORT_PREFIX}/lib/librmm.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
