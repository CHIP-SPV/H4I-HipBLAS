# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# Add the basic context tests
add_executable(hipblas_context_test context_test.cpp)
target_link_libraries(hipblas_context_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_context_test PRIVATE ${CMAKE_BINARY_DIR}/include)

# Add the simple batched test
add_executable(hipblas_simple_batched_test simple_batched_test.cpp)
target_link_libraries(hipblas_simple_batched_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_simple_batched_test PRIVATE ${CMAKE_BINARY_DIR}/include)

# Add the batched correctness tests
add_executable(hipblas_batched_correctness_test batched_correctness_test.cpp)
target_link_libraries(hipblas_batched_correctness_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_batched_correctness_test PRIVATE ${CMAKE_BINARY_DIR}/include)

# Add the strided-batched correctness tests
add_executable(hipblas_strided_correctness_test strided_correctness_test.cpp)
target_link_libraries(hipblas_strided_correctness_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_strided_correctness_test PRIVATE ${CMAKE_BINARY_DIR}/include)

# Add tests to CTest
add_test(NAME h4i-hipblas_context_test COMMAND hipblas_context_test)
add_test(NAME h4i-hipblas_simple_batched_test COMMAND hipblas_simple_batched_test)
add_test(NAME h4i-hipblas_batched_correctness_test COMMAND hipblas_batched_correctness_test)
add_test(NAME h4i-hipblas_strided_correctness_test COMMAND hipblas_strided_correctness_test)

# ILP64 (_64) Level 1 BLAS tests
add_executable(hipblas_blas1_64_test blas1_64_test.cpp)
target_link_libraries(hipblas_blas1_64_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_blas1_64_test PRIVATE ${CMAKE_BINARY_DIR}/include)
add_test(NAME h4i-hipblas_blas1_64_test COMMAND hipblas_blas1_64_test)

# GemmEx / DotEx mixed precision correctness tests.  Every GPU result is
# validated against classic CBLAS running on the CPU, so this test also needs
# the host MKL libraries.  MKL is already available here because
# MKLShimConfig.cmake pulls it in as a dependency.
#
# The sequential CPU layer is located and linked explicitly rather than via the
# umbrella MKL::MKL target: that target carries Intel OpenMP flags (-fiopenmp)
# which the plain clang++ used to build this project does not accept, and a
# scalar CPU reference has no use for threading anyway.
get_filename_component(H4I_MKL_LIB_DIR "${MKL_DIR}/../.." ABSOLUTE)
find_library(H4I_MKL_INTERFACE_LIB mkl_intel_ilp64 HINTS ${H4I_MKL_LIB_DIR} REQUIRED)
find_library(H4I_MKL_THREADING_LIB mkl_sequential   HINTS ${H4I_MKL_LIB_DIR} REQUIRED)
find_library(H4I_MKL_CORE_LIB      mkl_core         HINTS ${H4I_MKL_LIB_DIR} REQUIRED)

add_executable(hipblas_gemmex_dotex_correctness_test gemmex_dotex_correctness_test.cpp)
target_link_libraries(hipblas_gemmex_dotex_correctness_test PRIVATE hipblas hip::host
                      ${H4I_MKL_INTERFACE_LIB} ${H4I_MKL_THREADING_LIB} ${H4I_MKL_CORE_LIB})
target_compile_definitions(hipblas_gemmex_dotex_correctness_test PRIVATE MKL_ILP64)
target_include_directories(hipblas_gemmex_dotex_correctness_test PRIVATE
                           ${CMAKE_BINARY_DIR}/include ${MKL_INCLUDE})
add_test(NAME h4i-hipblas_gemmex_dotex_correctness_test COMMAND hipblas_gemmex_dotex_correctness_test)
