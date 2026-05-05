#=============================================================================
# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

#[=======================================================================[

Provide targets for the rapids-logger library.

rapids-logger provides an ABI stable interface to spdlog-like logging that can
be safely embedded into complex environments where exposing spdlog symbols or
having it as a public dependency makes stable environment difficult to create
or maintain safely.

Imported Targets
^^^^^^^^^^^^^^^^

If rapids_logger is found, this module defines the following IMPORTED GLOBAL
targets:

 rapids_logger::rapids_logger             - The rapids_logger library.

    

Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables::

  RAPIDS_LOGGER_FOUND
  RAPIDS_LOGGER_VERSION
  RAPIDS_LOGGER_VERSION_MAJOR
  RAPIDS_LOGGER_VERSION_MINOR

#]=======================================================================]


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

cmake_minimum_required(VERSION 3.26.4)

set(rapids_global_languages )
foreach(lang IN LISTS rapids_global_languages)
  include("${CMAKE_CURRENT_LIST_DIR}/rapids_logger-${lang}-language.cmake")
endforeach()
unset(rapids_global_languages)

include("${CMAKE_CURRENT_LIST_DIR}/rapids_logger-dependencies.cmake" OPTIONAL)
include("${CMAKE_CURRENT_LIST_DIR}/rapids_logger-targets.cmake" OPTIONAL)

if()
  set(rapids_logger_comp_names )
  # find dependencies before creating targets that use them
  # this way if a dependency can't be found we fail
  foreach(comp IN LISTS rapids_logger_FIND_COMPONENTS)
    if(${comp} IN_LIST rapids_logger_comp_names)
      file(GLOB rapids_logger_component_dep_files LIST_DIRECTORIES FALSE
           "${CMAKE_CURRENT_LIST_DIR}/rapids_logger-${comp}*-dependencies.cmake")
      foreach(f IN LISTS  rapids_logger_component_dep_files)
        include("${f}")
      endforeach()
    endif()
  endforeach()

  foreach(comp IN LISTS rapids_logger_FIND_COMPONENTS)
    if(${comp} IN_LIST rapids_logger_comp_names)
      file(GLOB rapids_logger_component_target_files LIST_DIRECTORIES FALSE
           "${CMAKE_CURRENT_LIST_DIR}/rapids_logger-${comp}*-targets.cmake")
      foreach(f IN LISTS  rapids_logger_component_target_files)
        include("${f}")
      endforeach()
      set(rapids_logger_${comp}_FOUND TRUE)
    endif()
  endforeach()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/rapids_logger-config-version.cmake" OPTIONAL)

# Set our version variables
set(RAPIDS_LOGGER_VERSION_MAJOR 0)
set(RAPIDS_LOGGER_VERSION_MINOR 1)
set(RAPIDS_LOGGER_VERSION_PATCH 0)
set(RAPIDS_LOGGER_VERSION 0.1.0)


set(rapids_global_targets rapids_logger)
set(rapids_namespaced_global_targets rapids_logger)
if((NOT "rapids_logger::" STREQUAL "") AND rapids_namespaced_global_targets)
  list(TRANSFORM rapids_namespaced_global_targets PREPEND "rapids_logger::")
endif()

foreach(target IN LISTS rapids_namespaced_global_targets)
  if(TARGET ${target})
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
      set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()

# For backwards compat
if("rapids_config_install" STREQUAL "rapids_config_build")
  foreach(target IN LISTS rapids_global_targets)
    if(TARGET ${target})
      get_target_property(_is_imported ${target} IMPORTED)
      get_target_property(_already_global ${target} IMPORTED_GLOBAL)
      if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
      if(NOT TARGET rapids_logger::${target})
        add_library(rapids_logger::${target} ALIAS ${target})
      endif()
    endif()
  endforeach()
endif()

unset(rapids_comp_names)
unset(rapids_comp_unique_ids)
unset(rapids_global_targets)
unset(rapids_namespaced_global_targets)

check_required_components(rapids_logger)

set(${CMAKE_FIND_PACKAGE_NAME}_CONFIG "${CMAKE_CURRENT_LIST_FILE}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME} CONFIG_MODE)

include("${CMAKE_CURRENT_LIST_DIR}/create_logger_macros.cmake")

