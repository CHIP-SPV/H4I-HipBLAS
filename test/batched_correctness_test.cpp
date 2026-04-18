#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "hipblas.h"
#include <complex>

#define CHECK_HIPBLAS_STATUS(status) \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "HipBLAS error at line " << __LINE__ << ": " << status << std::endl; \
        return false; \
    }

#define CHECK_HIP_STATUS(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP error at line " << __LINE__ << ": " << status << std::endl; \
        return false; \
    }

const double TOLERANCE_DOUBLE = 1e-8;
const float  TOLERANCE_FLOAT  = 1e-4f;

template<typename T>
bool compareArrays(const T* a, const T* b, int size, T tolerance) {
    for (int i = 0; i < size; i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] 
                      << " (diff: " << std::abs(a[i] - b[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
void applyRowPermutations(std::vector<T>& matrix, const int* pivots, int n, int lda);

// Helper function to generate random matrix in COLUMN-MAJOR storage
template<typename T>
void generateRandomMatrix(std::vector<T>& matrix, int rows, int cols, int lda, std::mt19937& gen) {
    std::uniform_real_distribution<typename std::conditional<std::is_same<T, float>::value, 
                                                           float, double>::type> dis(-1.0, 1.0);
    // Fill only the valid matrix elements in column-major order
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            matrix[col * lda + row] = static_cast<T>(dis(gen));
        }
    }
}

// Make matrix diagonally dominant to ensure non-singularity
template<typename T>
void makeDiagonallyDominant(std::vector<T>& matrix, int n, int lda) {
    for (int i = 0; i < n; i++) {
        T sum = 0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                sum += std::abs(matrix[j * lda + i]);
            }
        }
        matrix[i * lda + i] = sum + static_cast<T>(1.0);
    }
}

// CPU reference implementation for matrix multiplication (column-major)
template<typename T>
void cpu_gemm(bool transA, bool transB, int m, int n, int k,
              T alpha, const T* A, int lda, const T* B, int ldb,
              T beta, T* C, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            T sum = static_cast<T>(0);
            for (int l = 0; l < k; l++) {
                T a_val, b_val;
                if (transA) {
                    a_val = A[i * lda + l]; // A^T(i,l) = A(l,i)
                } else {
                    a_val = A[l * lda + i]; // A(i,l)
                }
                if (transB) {
                    b_val = B[l * ldb + j]; // B^T(l,j) = B(j,l)
                } else {
                    b_val = B[j * ldb + l]; // B(l,j)
                }
                sum += a_val * b_val;
            }
            // C(i,j) in column-major is at C[j*ldc + i]
            C[j * ldc + i] = alpha * sum + beta * C[j * ldc + i];
        }
    }
}

// Test double precision batched LU factorization
bool testDgetrfBatched() {
    std::cout << "Testing hipblasDgetrfBatched..." << std::endl;
    
    const int n = 4;
    const int lda = n;
    const int batchCount = 3;
    
    // Create hipBLAS handle
    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    
    // Allocate host memory
    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<int> ipiv_host(batchCount * n);
    std::vector<int> info_host(batchCount);
    
    // Generate random matrices with proper column-major layout
    std::mt19937 gen(42); // Fixed seed for reproducibility
    for (int b = 0; b < batchCount; b++) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
    }
    
    // Allocate device memory
    std::vector<double*> A_device_ptrs(batchCount);
    for (int b = 0; b < batchCount; b++) {
        CHECK_HIP_STATUS(hipMalloc(&A_device_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_device_ptrs[b], A_host[b].data(), 
                                   lda * n * sizeof(double), hipMemcpyHostToDevice));
    }
    
    double** A_device_array;
    int* ipiv_device;
    int* info_device;
    
    CHECK_HIP_STATUS(hipMalloc(&A_device_array, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv_device, batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info_device, batchCount * sizeof(int)));
    
    CHECK_HIP_STATUS(hipMemcpy(A_device_array, A_device_ptrs.data(), 
                               batchCount * sizeof(double*), hipMemcpyHostToDevice));
    
    // Call batched LU factorization
    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_device_array, lda, 
                                              ipiv_device, info_device, batchCount));
    
    // Copy results back
    CHECK_HIP_STATUS(hipMemcpy(ipiv_host.data(), ipiv_device, 
                               batchCount * n * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP_STATUS(hipMemcpy(info_host.data(), info_device, 
                               batchCount * sizeof(int), hipMemcpyDeviceToHost));
    
    // Verify all factorizations were successful
    bool success = true;
    for (int b = 0; b < batchCount; b++) {
        if (info_host[b] != 0) {
            std::cerr << "LU factorization failed for matrix " << b 
                      << " with info = " << info_host[b] << std::endl;
            success = false;
        }
    }
    
    // --- Numerical verification: check P*A == L*U ----------------------
    if (success) {
        std::vector<std::vector<double>> LU_host(batchCount, std::vector<double>(lda * n));
        // Copy factorised data back.
        for (int b = 0; b < batchCount; ++b) {
            CHECK_HIP_STATUS(hipMemcpy(LU_host[b].data(), A_device_ptrs[b],
                                       lda * n * sizeof(double), hipMemcpyDeviceToHost));

            // Build L and U matrices
            std::vector<double> L(n * n, 0.0), U(n * n, 0.0);
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < n; ++row) {
                    double val = LU_host[b][col * lda + row];
                    if (row > col) {
                        L[col * n + row] = val; // below diagonal
                    } else if (row == col) {
                        L[col * n + row] = 1.0;
                        U[col * n + row] = val;
                    } else {
                        U[col * n + row] = val; // above diagonal
                    }
                }
            }

            // Re-create P*A (apply row swaps to original)
            std::vector<double> P_A = A_host[b]; // original before factorisation
            applyRowPermutations(P_A, &ipiv_host[b * n], n, lda);

            // Compute L*U via reference GEMM
            std::vector<double> LU_recon(n * n, 0.0);
            cpu_gemm(false, false, n, n, n, 1.0, L.data(), n, U.data(), n, 0.0, LU_recon.data(), n);

            // Compare
            if (!compareArrays(P_A.data(), LU_recon.data(), n * n, TOLERANCE_DOUBLE)) {
                std::cerr << "LU factorisation contents wrong for batch " << b << std::endl;
                success = false;
            }
        }
    }
    
    // Cleanup
    for (int b = 0; b < batchCount; b++) {
        hipFree(A_device_ptrs[b]);
    }
    hipFree(A_device_array);
    hipFree(ipiv_device);
    hipFree(info_device);
    hipblasDestroy(handle);
    
    if (success) {
        std::cout << "hipblasDgetrfBatched test PASSED" << std::endl;
    }
    return success;
}

// Test double precision batched matrix inversion
bool testDgetriBatched() {
    std::cout << "Testing hipblasDgetriBatched..." << std::endl;
    
    const int n = 3;
    const int lda = n;
    const int ldc = n;
    const int batchCount = 2;
    
    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    
    // Allocate and initialize host memory
    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> C_host(batchCount, std::vector<double>(ldc * n));
    std::vector<int> ipiv_host(batchCount * n);
    std::vector<int> info_host(batchCount);
    
    std::mt19937 gen(123);
    for (int b = 0; b < batchCount; b++) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
        A_orig[b] = A_host[b]; // Keep original for verification
    }
    
    // Allocate device memory and copy data
    std::vector<double*> A_device_ptrs(batchCount);
    std::vector<double*> C_device_ptrs(batchCount);
    
    for (int b = 0; b < batchCount; b++) {
        CHECK_HIP_STATUS(hipMalloc(&A_device_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMalloc(&C_device_ptrs[b], ldc * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_device_ptrs[b], A_host[b].data(), 
                                   lda * n * sizeof(double), hipMemcpyHostToDevice));
    }
    
    double** A_device_array;
    double** C_device_array;
    int* ipiv_device;
    int* info_device;
    
    CHECK_HIP_STATUS(hipMalloc(&A_device_array, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&C_device_array, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv_device, batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info_device, batchCount * sizeof(int)));
    
    CHECK_HIP_STATUS(hipMemcpy(A_device_array, A_device_ptrs.data(), 
                               batchCount * sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_device_array, C_device_ptrs.data(), 
                               batchCount * sizeof(double*), hipMemcpyHostToDevice));
    
    // First factorize
    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_device_array, lda, 
                                              ipiv_device, info_device, batchCount));
    
    // Then invert
    CHECK_HIPBLAS_STATUS(hipblasDgetriBatched(handle, n, A_device_array, lda, 
                                              ipiv_device, C_device_array, ldc, 
                                              info_device, batchCount));
    
    // Copy results back
    CHECK_HIP_STATUS(hipMemcpy(info_host.data(), info_device, 
                               batchCount * sizeof(int), hipMemcpyDeviceToHost));
    for (int b = 0; b < batchCount; b++) {
        CHECK_HIP_STATUS(hipMemcpy(C_host[b].data(), C_device_ptrs[b], 
                                   ldc * n * sizeof(double), hipMemcpyDeviceToHost));
    }
    
    // Verify all inversions were successful
    bool success = true;
    for (int b = 0; b < batchCount; b++) {
        if (info_host[b] != 0) {
            std::cerr << "Matrix inversion failed for matrix " << b 
                      << " with info = " << info_host[b] << std::endl;
            success = false;
        }
    }
    
    // Verify correctness: A * A^(-1) should equal identity matrix
    if (success) {
        for (int b = 0; b < batchCount; b++) {
            std::vector<double> result(n * n, 0.0);
            // Compute A_orig * C using CPU reference (both in column-major)
            cpu_gemm(false, false, n, n, n, 1.0, A_orig[b].data(), lda, 
                     C_host[b].data(), ldc, 0.0, result.data(), n);
            
            // Check if result is close to identity matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    double actual = result[j * n + i]; // column-major indexing
                    if (std::abs(actual - expected) > TOLERANCE_DOUBLE) {
                        std::cerr << "Verification failed for batch " << b 
                                  << " at (" << i << "," << j << "): expected " 
                                  << expected << ", got " << actual << std::endl;
                        success = false;
                    }
                }
            }
        }
    }
    
    // Cleanup
    for (int b = 0; b < batchCount; b++) {
        hipFree(A_device_ptrs[b]);
        hipFree(C_device_ptrs[b]);
    }
    hipFree(A_device_array);
    hipFree(C_device_array);
    hipFree(ipiv_device);
    hipFree(info_device);
    hipblasDestroy(handle);
    
    if (success) {
        std::cout << "hipblasDgetriBatched test PASSED" << std::endl;
    }
    return success;
}

// Test single precision complex batched operations
bool testCgetrfBatched() {
    std::cout << "Testing hipblasCgetrfBatched..." << std::endl;
    
    const int n = 3;
    const int lda = n;
    const int batchCount = 2;
    
    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    
    // Create complex matrices (using hipblasComplex type)
    std::vector<std::vector<hipblasComplex>> A_host(batchCount, std::vector<hipblasComplex>(lda * n));
    std::vector<int> ipiv_host(batchCount * n);
    std::vector<int> info_host(batchCount);
    
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int b = 0; b < batchCount; b++) {
        // Fill matrix with random complex values in column-major order
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                A_host[b][col * lda + row] = {dis(gen), dis(gen)}; // real + imaginary parts
            }
        }
        
        // Make diagonally dominant using column-major indexing
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    // Column-major: element (i,j) is at index j*lda + i
                    sum += std::abs(A_host[b][j * lda + i].real()) + std::abs(A_host[b][j * lda + i].imag());
                }
            }
            // Diagonal element (i,i) is at index i*lda + i
            A_host[b][i * lda + i] = {sum + 1.0f, 0.0f};
        }
    }
    
    // Device setup
    std::vector<hipblasComplex*> A_device_ptrs(batchCount);
    for (int b = 0; b < batchCount; b++) {
        CHECK_HIP_STATUS(hipMalloc(&A_device_ptrs[b], lda * n * sizeof(hipblasComplex)));
        CHECK_HIP_STATUS(hipMemcpy(A_device_ptrs[b], A_host[b].data(), 
                                   lda * n * sizeof(hipblasComplex), hipMemcpyHostToDevice));
    }
    
    hipblasComplex** A_device_array;
    int* ipiv_device;
    int* info_device;
    
    CHECK_HIP_STATUS(hipMalloc(&A_device_array, batchCount * sizeof(hipblasComplex*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv_device, batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info_device, batchCount * sizeof(int)));
    
    CHECK_HIP_STATUS(hipMemcpy(A_device_array, A_device_ptrs.data(), 
                               batchCount * sizeof(hipblasComplex*), hipMemcpyHostToDevice));
    
    // Test the batched function
    CHECK_HIPBLAS_STATUS(hipblasCgetrfBatched(handle, n, A_device_array, lda, 
                                              ipiv_device, info_device, batchCount));
    
    // Check results
    CHECK_HIP_STATUS(hipMemcpy(info_host.data(), info_device, 
                               batchCount * sizeof(int), hipMemcpyDeviceToHost));
    
    bool success = true;
    for (int b = 0; b < batchCount; b++) {
        if (info_host[b] != 0) {
            std::cerr << "Complex LU factorization failed for matrix " << b 
                      << " with info = " << info_host[b] << std::endl;
            success = false;
        }
    }
    
    // Cleanup
    for (int b = 0; b < batchCount; b++) {
        hipFree(A_device_ptrs[b]);
    }
    hipFree(A_device_array);
    hipFree(ipiv_device);
    hipFree(info_device);
    hipblasDestroy(handle);
    
    if (success) {
        std::cout << "hipblasCgetrfBatched test PASSED" << std::endl;
    }
    return success;
}

// Test strided batched GEMM with proper column-major layout
bool testSgemmStridedBatched() {
    std::cout << "Testing hipblasSgemmStridedBatched..." << std::endl;
    
    const int m = 2, n = 2, k = 2;
    const int lda = m, ldb = k, ldc = m;
    const int batchCount = 3;
    const int strideA = lda * k;  // Column-major: m x k matrix needs lda * k storage
    const int strideB = ldb * n;  // Column-major: k x n matrix needs ldb * n storage  
    const int strideC = ldc * n;  // Column-major: m x n matrix needs ldc * n storage
    
    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    
    // Host memory
    std::vector<float> A_host(strideA * batchCount, 0.0f);
    std::vector<float> B_host(strideB * batchCount, 0.0f);
    std::vector<float> C_host(strideC * batchCount);
    std::vector<float> C_expected(strideC * batchCount, 0.0f);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Initialize matrices with simple known values for verification (column-major)
    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dis(0.1f, 1.0f);
    
    for (int batch = 0; batch < batchCount; batch++) {
        float* A_batch = &A_host[batch * strideA];
        float* B_batch = &B_host[batch * strideB];
        
        // Fill A matrix (m x k) in column-major order
        for (int col = 0; col < k; col++) {
            for (int row = 0; row < m; row++) {
                A_batch[col * lda + row] = dis(gen);
            }
        }
        
        // Fill B matrix (k x n) in column-major order
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < k; row++) {
                B_batch[col * ldb + row] = dis(gen);
            }
        }
        
        // Compute expected result using CPU reference
        float* C_batch = &C_expected[batch * strideC];
        cpu_gemm(false, false, m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_batch, ldc);
    }
    
    // Device memory
    float *A_device, *B_device, *C_device;
    CHECK_HIP_STATUS(hipMalloc(&A_device, strideA * batchCount * sizeof(float)));
    CHECK_HIP_STATUS(hipMalloc(&B_device, strideB * batchCount * sizeof(float)));
    CHECK_HIP_STATUS(hipMalloc(&C_device, strideC * batchCount * sizeof(float)));
    
    CHECK_HIP_STATUS(hipMemcpy(A_device, A_host.data(), 
                               strideA * batchCount * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_device, B_host.data(), 
                               strideB * batchCount * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemset(C_device, 0, strideC * batchCount * sizeof(float)));
    
    // Call strided batched GEMM
    CHECK_HIPBLAS_STATUS(hipblasSgemmStridedBatched(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                                    m, n, k, &alpha,
                                                    A_device, lda, strideA,
                                                    B_device, ldb, strideB,
                                                    &beta, C_device, ldc, strideC,
                                                    batchCount));
    
    // Copy result back
    CHECK_HIP_STATUS(hipMemcpy(C_host.data(), C_device, 
                               strideC * batchCount * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results against CPU reference
    bool success = compareArrays(C_host.data(), C_expected.data(), 
                                 strideC * batchCount, TOLERANCE_FLOAT);
    
    if (!success) {
        std::cerr << "Strided batched GEMM results do not match CPU reference" << std::endl;
    }
    
    // Cleanup
    hipFree(A_device);
    hipFree(B_device);
    hipFree(C_device);
    hipblasDestroy(handle);
    
    if (success) {
        std::cout << "hipblasSgemmStridedBatched test PASSED" << std::endl;
    }
    return success;
}

// Apply row interchanges stored in LAPACK-style pivot array to a column-major matrix.
template<typename T>
void applyRowPermutations(std::vector<T>& mat, const int* piv, int n, int lda)
{
    for (int i = 0; i < n; ++i) {
        int jp = piv[i] - 1;       // hipBLAS/LAPACK use 1-based indices.
        if (jp != i) {
            for (int col = 0; col < n; ++col) {
                std::swap(mat[col * lda + i], mat[col * lda + jp]);
            }
        }
    }
}

int main() {
    std::cout << "=== Testing H4I-HipBLAS Batched LAPACK Functions ===" << std::endl << std::endl;
    
    bool allPassed = true;
    
    // Probe fp64 support: Intel Arc (Alchemist) lacks native fp64 and MKL returns
    // HIPBLAS_STATUS_EXECUTION_FAILED for double ops on these devices.
    // H4I-HipBLAS catches the SYCL exception internally so it cannot be re-caught here;
    // use a dedicated probe call to detect the limitation before running the full tests.
    bool fp64_supported = true;
    {
        hipblasHandle_t probe_h = nullptr;
        if (hipblasCreate(&probe_h) == HIPBLAS_STATUS_SUCCESS) {
            double dummy = 0;
            double *d_dummy = nullptr;
            hipMalloc(&d_dummy, sizeof(double));
            hipblasStatus_t s = hipblasDasum(probe_h, 1, d_dummy, 1, &dummy);
            if (s == HIPBLAS_STATUS_EXECUTION_FAILED) {
                std::cout << "NOTE: device does not support fp64 — skipping D*/C* tests\n";
                fp64_supported = false;
            }
            hipFree(d_dummy);
            hipblasDestroy(probe_h);
        }
    }

    // Test all the newly implemented batched functions
    if (fp64_supported) {
        allPassed &= testDgetrfBatched();
        allPassed &= testDgetriBatched();
        allPassed &= testCgetrfBatched();
    } else {
        std::cout << "DgetrfBatched / DgetriBatched / CgetrfBatched SKIPPED (no fp64)\n";
    }
    allPassed &= testSgemmStridedBatched();
    
    std::cout << std::endl;
    if (allPassed) {
        std::cout << "ALL TESTS PASSED! Batched functions are working correctly." << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Some tests failed. Please check the implementation." << std::endl;
        return EXIT_FAILURE;
    }
} 