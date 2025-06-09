#include <iostream>
#include <hip/hip_runtime.h>
#include "hipblas.h"

int main() {
    std::cout << "Testing simple hipBLAS creation..." << std::endl;
    
    // Test context creation
    hipblasHandle_t handle = nullptr;
    hipblasStatus_t status = hipblasCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create hipBLAS handle: " << status << std::endl;
        return 1;
    }
    
    std::cout << "Created hipBLAS handle successfully" << std::endl;
    
    // Create simple device arrays to test the function signature
    const int n = 2;
    const int batchCount = 1;
    
    // Allocate a single matrix on device
    double* A_device;
    if (hipMalloc(&A_device, n * n * sizeof(double)) != hipSuccess) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        hipblasDestroy(handle);
        return 1;
    }
    
    // Create array of pointers on device
    double** A_array_device;
    if (hipMalloc(&A_array_device, batchCount * sizeof(double*)) != hipSuccess) {
        std::cerr << "Failed to allocate pointer array" << std::endl;
        hipFree(A_device);
        hipblasDestroy(handle);
        return 1;
    }
    
    // Copy the pointer to device
    if (hipMemcpy(A_array_device, &A_device, sizeof(double*), hipMemcpyHostToDevice) != hipSuccess) {
        std::cerr << "Failed to copy pointer to device" << std::endl;
        hipFree(A_array_device);
        hipFree(A_device);
        hipblasDestroy(handle);
        return 1;
    }
    
    // Allocate space for pivot array and info
    int* ipiv_device;
    int* info_device;
    if (hipMalloc(&ipiv_device, n * sizeof(int)) != hipSuccess ||
        hipMalloc(&info_device, sizeof(int)) != hipSuccess) {
        std::cerr << "Failed to allocate ipiv/info arrays" << std::endl;
        hipFree(A_array_device);
        hipFree(A_device);
        hipblasDestroy(handle);
        return 1;
    }
    
    std::cout << "Calling hipblasDgetrfBatched..." << std::endl;
    
    // Try calling the batched function
    status = hipblasDgetrfBatched(handle, n, A_array_device, n, ipiv_device, info_device, batchCount);
    
    if (status == HIPBLAS_STATUS_SUCCESS) {
        std::cout << "hipblasDgetrfBatched succeeded!" << std::endl;
    } else {
        std::cout << "hipblasDgetrfBatched failed with status: " << status << std::endl;
    }
    
    // Cleanup
    hipFree(info_device);
    hipFree(ipiv_device);
    hipFree(A_array_device);
    hipFree(A_device);
    
    status = hipblasDestroy(handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to destroy hipBLAS handle: " << status << std::endl;
        return 1;
    }
    
    std::cout << "Test completed!" << std::endl;
    return 0;
} 