# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-src")
  file(MAKE_DIRECTORY "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-src")
endif()
file(MAKE_DIRECTORY
  "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-build"
  "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix"
  "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix/tmp"
  "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix/src/rapids_logger-populate-stamp"
  "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix/src"
  "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix/src/rapids_logger-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix/src/rapids_logger-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py/_deps/rapids_logger-subbuild/rapids_logger-populate-prefix/src/rapids_logger-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
