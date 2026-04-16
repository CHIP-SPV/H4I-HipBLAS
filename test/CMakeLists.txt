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

# Add tests to CTest
add_test(NAME h4i-hipblas_context_test COMMAND hipblas_context_test)
add_test(NAME h4i-hipblas_simple_batched_test COMMAND hipblas_simple_batched_test)
add_test(NAME h4i-hipblas_batched_correctness_test COMMAND hipblas_batched_correctness_test)

# ILP64 (_64) Level 1 BLAS tests
add_executable(hipblas_blas1_64_test blas1_64_test.cpp)
target_link_libraries(hipblas_blas1_64_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_blas1_64_test PRIVATE ${CMAKE_BINARY_DIR}/include)
add_test(NAME h4i-hipblas_blas1_64_test COMMAND hipblas_blas1_64_test)
