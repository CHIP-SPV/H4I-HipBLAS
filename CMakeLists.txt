cmake_minimum_required(VERSION 3.24)
include (${CMAKE_SOURCE_DIR}/CMake/HipBLASVersion.cmake)
project(H4I-HipBLAS
    VERSION ${HipBLAS_VERSION}
    LANGUAGES CXX)

# We will use HIP in some fashion, no matter which platform
# we're targeting or what parts of the software we're building.
find_package(HIP REQUIRED)
# When using CHIP-SPV, CMake's method for finding a HIP compiler
# seems to want to find ROCm.  Since ROCm isn't in play, CMake
# fails to enable support for HIP as a first class language.
#enable_language(HIP)

# Define a target capturing common compiler configuration.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
# Unfortunately, it doesn't seem to be possible to set some
# of the C++ standards-related properties on this target
# and have it inherited by targets that "link" from it.
# We have to set the global variables that those target
# properties default to, or set them explicitly on the
# actual targets themselves with set_target_properties.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(H4I-HipBLASCommonConfig INTERFACE)
target_compile_features(H4I-HipBLASCommonConfig
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
option(H4I_BUILD_TESTS "Whether to build test programs." ON)
if(H4I_BUILD_TESTS)
    add_subdirectory(test)
endif()

