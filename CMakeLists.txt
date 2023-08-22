# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

cmake_minimum_required(VERSION 3.20)

# set the default CMAKE_INSTALL_PREFIX to the current directory/install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# print where MKLShim layer will be installed
message(STATUS "H4I-HipBLAS will be installed to: ${CMAKE_INSTALL_PREFIX}")

include (${CMAKE_CURRENT_SOURCE_DIR}/CMake/HipBLASVersion.cmake)
project(HipBLAS
    VERSION ${HipBLAS_VERSION}
    LANGUAGES CXX)

# check if CLANG_COMPILER_PATH was passed to CMake invocation, if not, look for clang++ in PATH
if(NOT DEFINED CLANG_COMPILER_PATH)
    find_program(CLANG_COMPILER_PATH clang++ REQUIRED)
    message(STATUS "Found LLVM C++ Compiler: ${CLANG_COMPILER_PATH}")
endif()

set(CMAKE_CXX_COMPILER ${CLANG_COMPILER_PATH})
set(CMAKE_CXX_COMPILER_ID "LLVM")

# We will use HIP in some fashion, no matter which platform
# we're targeting or what parts of the software we're building.
# Because of CMake's current assumption that HIP must require ROCm,
# we cannot use CMake's support for HIP as a first-class language.
# So we don't use enable_language() here, or specify it as a language
# in the project command.
#
# We want to find CHIP-SPV's HIP support, which uses 'hip' (lower case)
# as the name, not 'HIP' (upper case).
find_package(hip CONFIG REQUIRED)

# Ensure our users and installed tests will be able to find our
# dependency libraries.
include(GNUInstallDirs)
file(RELATIVE_PATH relRPath
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relRPath})

# Define a target capturing common configuration settings.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
# Unfortunately, it doesn't seem to be possible to set all
# of the properties we want on this target and have them
# be inherited by targets that "link" it.
# In particular, we can't set a few C++ standards properties
# and so either have to set them globally or on every target.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(HipBLASCommonConfig INTERFACE)
target_compile_features(HipBLASCommonConfig
    INTERFACE
        cxx_std_17
    )

# The primary purpose of the library is to provide a HipBLAS
# library that uses MKL as a backend.
# But we also support building the library's tests on a system
# with the ROCm HipBLAS installed, to enable comparisons
# between the behavior of this library and the ROCm implementation.
option(H4I_USE_ROCM_HIPBLAS "Whether to use ROCm-installed hipBLAS" OFF)
if(NOT H4I_USE_ROCM_HIPBLAS)
    # We want to use the H4I implementation.
    # Build our library.
    add_subdirectory(src)
endif()

# Build tests if desired.
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()

