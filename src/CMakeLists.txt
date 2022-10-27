
# We rely on the MKL shim library.
find_package(H4I-MKLShim REQUIRED)

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
add_library(H4I-HipBLAS SHARED
    sgemm.cpp)
target_include_directories(H4I-HipBLAS
	PRIVATE
		${CMAKE_SOURCE_DIR}/include
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
		$<INSTALL_INTERFACE:include>)

target_link_libraries(H4I-HipBLAS
	PRIVATE
        H4I-HipBLASCommonConfig

    PUBLIC
		hip::host
        H4I::MKLShim
    )

# Specify how to install the library.
include (GNUInstallDirs)
install(TARGETS H4I-HipBLAS
		EXPORT HipBLAS
)
install(FILES
			${CMAKE_BINARY_DIR}/include/hipblas.h
			${CMAKE_BINARY_DIR}/include/hipblas-version.h
			${CMAKE_SOURCE_DIR}/include/hipblas-export.h
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
)

install(EXPORT HipBLAS
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/H4I-HipBLAS
	NAMESPACE H4I::
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/CMake/H4I-HipBLASConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/H4I-HipBLASConfig.cmake
	PATH_VARS CMAKE_INSTALL_INCLUDEDIR
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/H4I-HipBLAS
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/H4I-HipBLASConfig.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/H4I-HipBLAS
)

