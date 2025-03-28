# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# Add the basic context tests
add_executable(hipblas_context_test context_test.cpp)
target_link_libraries(hipblas_context_test PRIVATE hipblas hip::host)
target_include_directories(hipblas_context_test PRIVATE ${CMAKE_BINARY_DIR}/include)

# Add tests to CTest
add_test(NAME h4i-hipblas_context_test COMMAND hipblas_context_test)

