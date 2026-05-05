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

include(CMakeFindDependencyMacro)

#=============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

#[=======================================================================[.rst:
rapids_cmake_download_with_retry
--------------------------------

.. versionadded:: v25.06.00

Downloads a file from a URL with retry logic for handling network issues.

  .. code-block:: cmake

    rapids_cmake_download_with_retry(url output_file sha256 [MAX_RETRIES <max_retries>] [RETRY_DELAY <retry_delay>])

This function will attempt to download the file multiple times if network issues occur.
It verifies the download by checking the SHA256 checksum of the downloaded file. If all
retries fail, it will raise a fatal error.

``url``
  The URL to download from.

``output_file``
  The path where the downloaded file should be saved.

``sha256``
  The expected SHA256 checksum of the file.

``MAX_RETRIES``
  Maximum number of retry attempts. Defaults to 10.

``RETRY_DELAY``
  Delay between retries in seconds. Defaults to 5.

#]=======================================================================]
function(rapids_cmake_download_with_retry url output_file sha256)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.download_with_retry")

  set(options)
  set(one_value MAX_RETRIES RETRY_DELAY)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Set default values for optional arguments
  if(NOT DEFINED _RAPIDS_MAX_RETRIES)
    set(_RAPIDS_MAX_RETRIES 10)
  endif()
  if(NOT DEFINED _RAPIDS_RETRY_DELAY)
    set(_RAPIDS_RETRY_DELAY 5)
  endif()

  # Set up retry parameters
  set(current_retry 0)
  set(download_success FALSE)

  while(NOT download_success AND current_retry LESS ${_RAPIDS_MAX_RETRIES})
    if(current_retry GREATER 0)
      message(STATUS "Retrying download (attempt ${current_retry} of ${_RAPIDS_MAX_RETRIES}) after ${_RAPIDS_RETRY_DELAY} seconds..."
      )
      execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${_RAPIDS_RETRY_DELAY})
    endif()

    # Remove any existing file to ensure clean download
    if(EXISTS "${output_file}")
      file(REMOVE "${output_file}")
    endif()

    file(DOWNLOAD "${url}" "${output_file}" LOG download_log)

    # Check if file exists and validate SHA256
    if(EXISTS "${output_file}")
      file(SHA256 "${output_file}" downloaded_sha256)
      if(downloaded_sha256 STREQUAL "${sha256}")
        set(download_success TRUE)
      else()
        message(WARNING "Downloaded file SHA256 checksum mismatch. Expected: ${sha256}, Got: ${downloaded_sha256}"
        )
        file(REMOVE "${output_file}")
      endif()
    endif()

    if(NOT download_success)
      math(EXPR current_retry "${current_retry} + 1")
      if(current_retry LESS ${_RAPIDS_MAX_RETRIES})
        message(WARNING "Failed to download file. Will retry. Download log:\n${download_log}")
      else()
        message(FATAL_ERROR "Failed to download file after ${_RAPIDS_MAX_RETRIES} attempts. Download log:\n${download_log}"
        )
      endif()
    endif()
  endwhile()
endfunction()
#=============================================================================
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

#[=======================================================================[.rst:
rapids_cpm_download
-------------------

.. versionadded:: v21.10.00

Does the downloading of the `CPM` module

.. code-block:: cmake

  rapids_cpm_download()

The CPM module will be downloaded based on the following.

.. versionadded:: v24.10.00

If :cmake:variable:`CPM_DOWNLOAD_LOCATION` is defined that location will be used
as the download location. If a file already exists at that location no download will occur

If the :cmake:variable:`CPM_SOURCE_CACHE` or :cmake:variable:`ENV{CPM_SOURCE_CACHE}` are
defined those will be used to compute a location for the file.

If none of the above variables are defined, rapids-cmake will download the file
to `cmake` directory under :cmake:variable:`CMAKE_BINARY_DIR`.

.. note::
  Use `rapids_cpm_init` instead of this function, as this is an implementation detail
  required for proper cpm project exporting in build directories

  This function can't call other rapids-cmake functions, due to the
  restrictions of `write_dependencies.cmake`

#]=======================================================================]
function(rapids_cpm_download)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.download")

  # When changing version verify no new variables needs to be propagated
  set(CPM_DOWNLOAD_VERSION 0.40.0)
  set(CPM_DOWNLOAD_SHA256_HASH 7b354f3a5976c4626c876850c93944e52c83ec59a159ae5de5be7983f0e17a2a)

  if(NOT DEFINED CPM_DOWNLOAD_LOCATION)
    if(CPM_SOURCE_CACHE)
      # Expand relative path. This is important if the provided path contains a tilde (~)
      cmake_path(ABSOLUTE_PATH CPM_SOURCE_CACHE)

      # default to the same location that cpm computes
      set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      if(EXISTS "${CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        # Also support the rapids-cmake download location ( cmake/ vs cpm/ )
        set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      endif()

    elseif(DEFINED ENV{CPM_SOURCE_CACHE})

      # default to the same location that cpm computes
      set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      if(EXISTS "$ENV{CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        # Also support the rapids-cmake download location ( cmake/ vs cpm/ )
        set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      endif()

    else()
      set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    endif()
  endif()

  if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(VERBOSE "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    if(NOT COMMAND rapids_cmake_download_with_retry)
      include("${rapids-cmake-dir}/cmake/download_with_retry.cmake")
    endif()
    rapids_cmake_download_with_retry(
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
      ${CPM_DOWNLOAD_LOCATION} ${CPM_DOWNLOAD_SHA256_HASH})
  endif()

  include(${CPM_DOWNLOAD_LOCATION})

  # Propagate up any modified local variables that CPM has changed.
  #
  # Push up the modified CMAKE_MODULE_PATh to allow `find_package` calls to find packages that CPM
  # already added.
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

endfunction()
rapids_cpm_download()

# re-use our CPM source cache if not set
if(NOT DEFINED CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE "@CPM_SOURCE_CACHE@")
  set(rapids_clear_cpm_cache true)
endif()
#=============================================================================
# find_dependency Search for CUDAToolkit
#
# Make sure we search for a build-dir config module for the project
set(possible_package_dir "")
if(possible_package_dir AND NOT DEFINED CUDAToolkit_DIR)
  set(CUDAToolkit_DIR "${possible_package_dir}")
endif()

find_dependency(CUDAToolkit)

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================

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


# CPM Search for rapids_logger
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/usr/local/lib/cmake/rapids_logger")
if(possible_package_dir AND NOT DEFINED rapids_logger_DIR)
  set(rapids_logger_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;rapids_logger;VERSION;0.1.0;GIT_REPOSITORY;https://github.com/rapidsai/rapids-logger.git;GIT_TAG;46070bb255482f0782ca840ae45de9354380e298;GIT_SHALLOW;OFF;OPTIONS;BUILD_TESTS OFF"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================

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


# CPM Search for CCCL
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "")
if(possible_package_dir AND NOT DEFINED CCCL_DIR)
  set(CCCL_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;CCCL;VERSION;3.0.3;FIND_PACKAGE_ARGUMENTS;EXACT;GIT_REPOSITORY;https://github.com/NVIDIA/cccl.git;GIT_TAG;8c04b6539859932f5602e86d38314e4d87f96420;GIT_SHALLOW;OFF;EXCLUDE_FROM_ALL;OFF"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================

if(CCCL_FOUND)
    target_compile_definitions(CCCL::CCCL INTERFACE CUB_DISABLE_NAMESPACE_MAGIC)
    target_compile_definitions(CCCL::CCCL INTERFACE CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_DISABLE_ABI_NAMESPACE)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
    target_compile_definitions(CCCL::CCCL INTERFACE CCCL_DISABLE_PDL)
    

endif()
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


# CPM Search for nvtx3
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/usr/local/lib/cmake/nvtx3")
if(possible_package_dir AND NOT DEFINED nvtx3_DIR)
  set(nvtx3_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;nvtx3;VERSION;3.2.0;GIT_REPOSITORY;https://github.com/NVIDIA/NVTX.git;GIT_TAG;69c9949150ac1c310758a304082228a36d5e4758;GIT_SHALLOW;OFF;SOURCE_SUBDIR;c;EXCLUDE_FROM_ALL;OFF;OPTIONS;NVTX3_INSTALL ON"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================


set(rapids_global_targets CCCL;CCCL::CCCL;CCCL::CUB;CCCL::libcudacxx;nvtx3-c;nvtx3-cpp)


foreach(target IN LISTS rapids_global_targets)
  if(TARGET ${target})
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()

unset(rapids_global_targets)
unset(rapids_clear_cpm_cache)
