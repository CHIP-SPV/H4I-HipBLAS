cmake_minimum_required(VERSION 3.15)

# Add basic test
add_executable(blas_basic_test blas_basic_test.cpp)
if(TARGET deviceInternal) # deviceInternal is available when building in-tree
    target_link_libraries(blas_basic_test PRIVATE deviceInternal hipblas)
else()
    target_link_libraries(blas_basic_test PRIVATE hip::device hipblas)
endif()

# Install the sample in bin directory
install(TARGETS blas_basic_test DESTINATION bin)

# Add the test to CTest
add_test(NAME h4i-hipblas_basic_context_test COMMAND blas_basic_test) 