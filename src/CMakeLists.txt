# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# We rely on the MKL shim library.
find_package(MKLShim REQUIRED)

# To aid portability between our implementation and
# the ROCm hipBLAS implementation, we use the ROCm
# hipBLAS library's hipblas.h header.
include(ExternalProject)
option(H4I_ROCM_HIPBLAS_TAG "Tag to use from the ROCm hipBLAS repository when obtaining its hipblas.h header" "rocm-5.3.0")
ExternalProject_Add(ROCmHipblas
    GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/hipblas
    GIT_TAG ${H4I_ROCM_HIPBLAS_TAG}

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND sh ${CMAKE_SOURCE_DIR}/Scripts/install-hipblas-header.sh <SOURCE_DIR> ${CMAKE_BINARY_DIR} ${CMAKE_INSTALL_PREFIX}
)

# Product the hipblas version header required by hipblas.h in recent ROCm versions.
configure_file(
    ${CMAKE_SOURCE_DIR}/include/hipblas-version.h.in
    ${CMAKE_BINARY_DIR}/include/hipblas-version.h
    @ONLY)

# Specify how to build the library.
add_library(hipblas SHARED
    util.cpp
    hipblas.cpp)
target_include_directories(hipblas
	PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include;${CMAKE_BINARY_DIR}/include>"
		$<INSTALL_INTERFACE:include>)
# We need to be sure the ROCm HipBLAS library is downloaded and "installed"
# (i.e., the hipblas.h header has been copied to the location we want) before
# we build this library.
add_dependencies(hipblas ROCmHipblas)

target_link_libraries(hipblas
	PRIVATE
        HipBLASCommonConfig
        H4I::MKLShim
    PUBLIC
		hip::host
    )


# Specify how to install the library.
include (GNUInstallDirs)
install(TARGETS hipblas
        EXPORT hipblas
)
install(FILES
			${CMAKE_BINARY_DIR}/include/hipblas.h
			${CMAKE_BINARY_DIR}/include/hipblas-version.h
			${CMAKE_SOURCE_DIR}/include/hipblas-export.h
        TYPE INCLUDE
)

install(EXPORT hipblas
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblas
	NAMESPACE H4I::
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/CMake/HipBLASConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/hipblasConfig.cmake
	PATH_VARS CMAKE_INSTALL_INCLUDEDIR
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblas
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/hipblasConfig.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblas
)

