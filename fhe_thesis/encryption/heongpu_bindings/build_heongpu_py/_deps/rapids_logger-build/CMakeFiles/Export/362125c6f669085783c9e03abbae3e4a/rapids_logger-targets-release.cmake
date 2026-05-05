#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rapids_logger::rapids_logger" for configuration "Release"
set_property(TARGET rapids_logger::rapids_logger APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rapids_logger::rapids_logger PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librapids_logger.so"
  IMPORTED_SONAME_RELEASE "librapids_logger.so"
  )

list(APPEND _cmake_import_check_targets rapids_logger::rapids_logger )
list(APPEND _cmake_import_check_files_for_rapids_logger::rapids_logger "${_IMPORT_PREFIX}/lib/librapids_logger.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
