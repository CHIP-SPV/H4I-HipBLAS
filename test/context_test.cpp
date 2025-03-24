// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.

#include <iostream>
#include <hip/hip_runtime.h>
#include "hipblas.h"

// Simple macro to check hipBLAS status
#define CHECK_HIPBLAS_STATUS(status) \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "HipBLAS error at line " << __LINE__ << ": " << status << std::endl; \
        return EXIT_FAILURE; \
    }

int main() {
    std::cout << "Testing HipBLAS Context Creation and Basic Functions" << std::endl;
    
    // Test context creation
    hipblasHandle_t handle = nullptr;
    hipblasStatus_t status = hipblasCreate(&handle);
    CHECK_HIPBLAS_STATUS(status);
    
    if (handle == nullptr) {
        std::cerr << "Failed to create HipBLAS handle" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Successfully created HipBLAS handle" << std::endl;
    
    // Test setting pointer mode
    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    CHECK_HIPBLAS_STATUS(status);
    
    // Test getting pointer mode
    hipblasPointerMode_t mode;
    status = hipblasGetPointerMode(handle, &mode);
    CHECK_HIPBLAS_STATUS(status);
    
    if (mode != HIPBLAS_POINTER_MODE_HOST) {
        std::cerr << "Unexpected pointer mode value" << std::endl;
        hipblasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    std::cout << "Pointer mode successfully set and retrieved" << std::endl;
    
    // Test setting stream
    hipStream_t stream;
    hipError_t hipStatus = hipStreamCreate(&stream);
    if (hipStatus != hipSuccess) {
        std::cerr << "Failed to create HIP stream" << std::endl;
        hipblasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    status = hipblasSetStream(handle, stream);
    CHECK_HIPBLAS_STATUS(status);
    std::cout << "Successfully set stream" << std::endl;
    
    // Clean up
    hipStreamDestroy(stream);
    
    // Test destruction
    status = hipblasDestroy(handle);
    CHECK_HIPBLAS_STATUS(status);
    
    std::cout << "All tests passed successfully!" << std::endl;
    return EXIT_SUCCESS;
} 