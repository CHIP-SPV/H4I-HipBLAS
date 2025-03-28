#include <iostream>
#include <vector>
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "hipblas.h"

#define HIP_CHECK(stat)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = stat;                                          \
        if(err != hipSuccess)                                           \
        {                                                               \
            std::cerr << "HIP error: " << hipGetErrorString(err)        \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            exit(err);                                                  \
        }                                                               \
    } while(0)

#define HIPBLAS_CHECK(stat)                                             \
    do                                                                  \
    {                                                                   \
        hipblasStatus_t err = stat;                                     \
        if(err != HIPBLAS_STATUS_SUCCESS)                               \
        {                                                               \
            std::cerr << "hipBLAS error: " << err                       \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            exit(err);                                                  \
        }                                                               \
    } while(0)

int main() {
    std::cout << "======== H4I-HipBLAS Basic Test ========" << std::endl;
    
    // Test context creation
    std::cout << "Testing hipBLAS context creation..." << std::endl;
    hipblasHandle_t handle = nullptr;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    if (handle) {
        std::cout << "Context creation successful!" << std::endl;
    } else {
        std::cerr << "Failed to create context!" << std::endl;
        return -1;
    }

    // Test stream creation and setting
    std::cout << "Testing hipBLAS stream handling..." << std::endl;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
    std::cout << "Stream handling successful!" << std::endl;

    // Test a simple BLAS operation (SASUM - sum of absolute values)
    std::cout << "Testing hipBLAS operation (sasum)..." << std::endl;
    const int n = 5;
    std::vector<float> hx = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    float* dx;
    float result;
    float expected_result = 15.0f; // |1| + |-2| + |3| + |-4| + |5| = 15

    HIP_CHECK(hipMalloc(&dx, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, hx.data(), n * sizeof(float), hipMemcpyHostToDevice));

    HIPBLAS_CHECK(hipblasSasum(handle, n, dx, 1, &result));
    HIP_CHECK(hipDeviceSynchronize());
    
    std::cout << "SASUM result: " << result << " (expected: " << expected_result << ")" << std::endl;
    if (std::abs(result - expected_result) < 1e-5) {
        std::cout << "BLAS operation successful!" << std::endl;
    } else {
        std::cerr << "BLAS operation result mismatch!" << std::endl;
    }

    // Clean up
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipStreamDestroy(stream));
    HIPBLAS_CHECK(hipblasDestroy(handle));
    
    std::cout << "======== Test Complete ========" << std::endl;
    return 0;
} 