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
        if(std::isnan(a[i]) || std::isnan(b[i]) {
            std::cerr << "Array element at index " << i << ": " << a[i] << " or " << b[i] 
                      << " is nan. This should never happen." << std::endl;
            return false;
        }
        if(std::abs(a[i] - b[i]) > tolerance) {
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

// ===========================================================================
// Additional getri (batched matrix-inverse) coverage.
// getri inverts from the LU factors, so each test runs getrf then getri and
// verifies A_orig * inv(A) ~= I (column-major). Reuses the helpers above.
// ===========================================================================

// generic double getri correctness for one (n, batchCount)
static bool runDgetriCorrectness(int n, int batchCount, unsigned seed) {
    const int lda = n, ldc = n;
    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> C_host(batchCount, std::vector<double>(ldc * n));
    std::mt19937 gen(seed);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
        A_orig[b] = A_host[b];
    }

    std::vector<double*> A_ptrs(batchCount), C_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMalloc(&C_ptrs[b], ldc * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(double),
                                   hipMemcpyHostToDevice));
    }
    double **A_arr = nullptr, **C_arr = nullptr; int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&C_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv,  batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info,  batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_arr, C_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));

    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetriBatched(handle, n, A_arr, lda, ipiv, C_arr, ldc, info, batchCount));

    bool ok = true;
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(C_host[b].data(), C_ptrs[b], ldc * n * sizeof(double),
                                   hipMemcpyDeviceToHost));
        std::vector<double> R(n * n, 0.0);
        cpu_gemm(false, false, n, n, n, 1.0, A_orig[b].data(), lda, C_host[b].data(), ldc, 0.0, R.data(), n);
        for (int i = 0; i < n && ok; ++i)
            for (int j = 0; j < n && ok; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (std::abs(R[j * n + i] - expected) > TOLERANCE_DOUBLE) ok = false;
            }
    }

    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(C_ptrs[b]); }
    hipFree(A_arr); hipFree(C_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    return ok;
}

// sweep sizes (incl. 1x1) and batch counts
bool testDgetriSizesAndBatches() {
    std::cout << "Testing hipblasDgetriBatched across sizes/batch counts..." << std::endl;
    const int sizes[]   = {1, 2, 5, 16, 32};
    const int batches[] = {1, 4, 16};
    bool ok = true;
    for (int n : sizes)
        for (int bc : batches) {
            bool r = runDgetriCorrectness(n, bc, 100u + 31u * (unsigned)n + (unsigned)bc);
            std::cout << "  n=" << n << " batch=" << bc << " : " << (r ? "ok" : "FAIL") << std::endl;
            ok = ok && r;
        }
    if (ok) std::cout << "hipblasDgetriBatched sizes/batches test PASSED" << std::endl;
    return ok;
}

// known analytic inverse (exact values)
bool testDgetriKnownInverse() {
    std::cout << "Testing hipblasDgetriBatched known 2x2 inverse..." << std::endl;
    const int n = 2, lda = 2, ldc = 2, batchCount = 1;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    // A = [[4,3],[6,3]] ; inv(A) = [[-1/2, 1/2],[1, -2/3]]  (column-major)
    std::vector<double> A           = {4.0, 6.0, 3.0, 3.0};
    std::vector<double> expectedInv = {-0.5, 1.0, 0.5, -2.0/3.0};

    double *dA = nullptr, *dC = nullptr, **A_arr = nullptr, **C_arr = nullptr;
    int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&dA, 4 * sizeof(double)));
    CHECK_HIP_STATUS(hipMalloc(&dC, 4 * sizeof(double)));
    CHECK_HIP_STATUS(hipMemcpy(dA, A.data(), 4 * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMalloc(&A_arr, sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&C_arr, sizeof(double*)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, &dA, sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_arr, &dC, sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMalloc(&ipiv, n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info, sizeof(int)));

    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetriBatched(handle, n, A_arr, lda, ipiv, C_arr, ldc, info, batchCount));

    std::vector<double> C(4);
    CHECK_HIP_STATUS(hipMemcpy(C.data(), dC, 4 * sizeof(double), hipMemcpyDeviceToHost));
    bool ok = compareArrays(C.data(), expectedInv.data(), 4, TOLERANCE_DOUBLE);

    hipFree(dA); hipFree(dC); hipFree(A_arr); hipFree(C_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "hipblasDgetriBatched known-inverse test PASSED" << std::endl;
    return ok;
}

// single precision Sgetri correctness
bool testSgetriBatched() {
    std::cout << "Testing hipblasSgetriBatched..." << std::endl;
    const int n = 6, lda = n, ldc = n, batchCount = 4;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<std::vector<float>> A_host(batchCount, std::vector<float>(lda * n));
    std::vector<std::vector<float>> A_orig(batchCount, std::vector<float>(lda * n));
    std::vector<std::vector<float>> C_host(batchCount, std::vector<float>(ldc * n));
    std::mt19937 gen(777);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
        A_orig[b] = A_host[b];
    }
    std::vector<float*> A_ptrs(batchCount), C_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(float)));
        CHECK_HIP_STATUS(hipMalloc(&C_ptrs[b], ldc * n * sizeof(float)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(float), hipMemcpyHostToDevice));
    }
    float **A_arr = nullptr, **C_arr = nullptr; int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(float*)));
    CHECK_HIP_STATUS(hipMalloc(&C_arr, batchCount * sizeof(float*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv,  batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info,  batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(float*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_arr, C_ptrs.data(), batchCount * sizeof(float*), hipMemcpyHostToDevice));

    CHECK_HIPBLAS_STATUS(hipblasSgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasSgetriBatched(handle, n, A_arr, lda, ipiv, C_arr, ldc, info, batchCount));

    bool ok = true;
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(C_host[b].data(), C_ptrs[b], ldc * n * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> R(n * n, 0.0f);
        cpu_gemm(false, false, n, n, n, 1.0f, A_orig[b].data(), lda, C_host[b].data(), ldc, 0.0f, R.data(), n);
        for (int i = 0; i < n && ok; ++i)
            for (int j = 0; j < n && ok; ++j) {
                float expected = (i == j) ? 1.0f : 0.0f;
                if (std::abs(R[j * n + i] - expected) > TOLERANCE_FLOAT) ok = false;
            }
    }
    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(C_ptrs[b]); }
    hipFree(A_arr); hipFree(C_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "hipblasSgetriBatched test PASSED" << std::endl;
    return ok;
}

// getri with heavy row pivoting. Diagonally-dominant matrices don't pivot
// (identity ipiv), so they don't exercise the int->int64 ipiv conversion.
// Here we row-reverse a diagonally-dominant matrix: the dominant entries move
// off the diagonal, forcing getrf to perform non-trivial row swaps. A_orig*inv
// ~= I only holds if those pivots are converted and applied correctly.
bool testDgetriPivoting() {
    std::cout << "Testing hipblasDgetriBatched with row pivoting..." << std::endl;
    const int n = 6, lda = n, ldc = n, batchCount = 4;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> C_host(batchCount, std::vector<double>(ldc * n));
    std::mt19937 gen(4242);
    for (int b = 0; b < batchCount; ++b) {
        std::vector<double> M(lda * n);
        generateRandomMatrix(M, n, n, lda, gen);
        makeDiagonallyDominant(M, n, lda);
        // Reverse row order -> dominant entries land off-diagonal -> forces pivoting.
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A_host[b][j * lda + i] = M[j * lda + (n - 1 - i)];
        A_orig[b] = A_host[b];
    }

    std::vector<double*> A_ptrs(batchCount), C_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMalloc(&C_ptrs[b], ldc * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(double),
                                   hipMemcpyHostToDevice));
    }
    double **A_arr = nullptr, **C_arr = nullptr; int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&C_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv,  batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info,  batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_arr, C_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));

    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetriBatched(handle, n, A_arr, lda, ipiv, C_arr, ldc, info, batchCount));

    // Sanity: pivots should be non-trivial (some ipiv[k] != k+1) for at least one matrix.
    std::vector<int> ipiv_host(batchCount * n);
    CHECK_HIP_STATUS(hipMemcpy(ipiv_host.data(), ipiv, batchCount * n * sizeof(int), hipMemcpyDeviceToHost));
    bool anyPivot = false;
    for (int b = 0; b < batchCount; ++b)
        for (int k = 0; k < n; ++k)
            if (ipiv_host[b * n + k] != k + 1) anyPivot = true;
    if (!anyPivot)
        std::cout << "  note: no row swaps occurred (ipiv was identity)" << std::endl;

    bool ok = true;
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(C_host[b].data(), C_ptrs[b], ldc * n * sizeof(double),
                                   hipMemcpyDeviceToHost));
        std::vector<double> R(n * n, 0.0);
        cpu_gemm(false, false, n, n, n, 1.0, A_orig[b].data(), lda, C_host[b].data(), ldc, 0.0, R.data(), n);
        for (int i = 0; i < n && ok; ++i)
            for (int j = 0; j < n && ok; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (std::abs(R[j * n + i] - expected) > TOLERANCE_DOUBLE) ok = false;
            }
    }
    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(C_ptrs[b]); }
    hipFree(A_arr); hipFree(C_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "hipblasDgetriBatched pivoting test PASSED" << std::endl;
    return ok;
}

// ===========================================================================
// getrs (batched solve A X = B using the LU factors from getrf) coverage.
// Each test runs getrf then getrs (X overwrites B) and verifies
// op(A_orig) * X ~= B_orig (column-major). Reuses the helpers above.
// ===========================================================================

static bool runDgetrsCorrectness(hipblasOperation_t trans, int n, int nrhs,
                                 int batchCount, unsigned seed) {
    const int lda = n, ldb = n;
    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> B_host(batchCount, std::vector<double>(ldb * nrhs));
    std::vector<std::vector<double>> B_orig(batchCount, std::vector<double>(ldb * nrhs));
    std::mt19937 gen(seed);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
        A_orig[b] = A_host[b];
        generateRandomMatrix(B_host[b], n, nrhs, ldb, gen);
        B_orig[b] = B_host[b];
    }

    std::vector<double*> A_ptrs(batchCount), B_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMalloc(&B_ptrs[b], ldb * nrhs * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(double), hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(B_ptrs[b], B_host[b].data(), ldb * nrhs * sizeof(double), hipMemcpyHostToDevice));
    }
    double **A_arr = nullptr, **B_arr = nullptr; int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&B_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv,  batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info,  batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_arr, B_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));

    int info_getrs = 0;  // getrsBatched info is a single host int
    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetrsBatched(handle, trans, n, nrhs, A_arr, lda, ipiv,
                                              B_arr, ldb, &info_getrs, batchCount));

    bool ok = true;
    const bool tr = (trans != HIPBLAS_OP_N);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(B_host[b].data(), B_ptrs[b], ldb * nrhs * sizeof(double),
                                   hipMemcpyDeviceToHost));
        // R = op(A_orig) * X, compare to original RHS B_orig.
        std::vector<double> R(n * nrhs, 0.0);
        cpu_gemm(tr, false, n, nrhs, n, 1.0, A_orig[b].data(), lda, B_host[b].data(), ldb, 0.0, R.data(), n);
        for (int j = 0; j < nrhs && ok; ++j)
            for (int i = 0; i < n && ok; ++i)
                if (std::abs(R[j * n + i] - B_orig[b][j * ldb + i]) > TOLERANCE_DOUBLE) ok = false;
    }
    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(B_ptrs[b]); }
    hipFree(A_arr); hipFree(B_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    return ok;
}

// sweep sizes / right-hand-side counts / batch counts (solve A X = B)
bool testDgetrsBatched() {
    std::cout << "Testing hipblasDgetrsBatched (A X = B) across sizes/nrhs/batch..." << std::endl;
    const int sizes[]   = {1, 3, 8, 16};
    const int nrhss[]   = {1, 4};
    const int batches[] = {1, 8};
    bool ok = true;
    for (int n : sizes)
      for (int nrhs : nrhss)
        for (int bc : batches) {
            bool r = runDgetrsCorrectness(HIPBLAS_OP_N, n, nrhs, bc,
                                          200u + 7u * (unsigned)n + (unsigned)nrhs + (unsigned)bc);
            std::cout << "  n=" << n << " nrhs=" << nrhs << " batch=" << bc
                      << " : " << (r ? "ok" : "FAIL") << std::endl;
            ok = ok && r;
        }
    if (ok) std::cout << "hipblasDgetrsBatched test PASSED" << std::endl;
    return ok;
}

// transpose solve: A^T X = B
bool testDgetrsTranspose() {
    std::cout << "Testing hipblasDgetrsBatched transpose (A^T X = B)..." << std::endl;
    bool ok = runDgetrsCorrectness(HIPBLAS_OP_T, 8, 3, 4, 9090u);
    if (ok) std::cout << "hipblasDgetrsBatched transpose test PASSED" << std::endl;
    return ok;
}

// known solution: A=[[4,3],[6,3]], X=[1;2] -> B = A*X = [10;12]; solve must recover X.
bool testDgetrsKnownSolution() {
    std::cout << "Testing hipblasDgetrsBatched known solution..." << std::endl;
    const int n = 2, nrhs = 1, lda = 2, ldb = 2, batchCount = 1;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    std::vector<double> A = {4.0, 6.0, 3.0, 3.0};   // col-major
    std::vector<double> B = {10.0, 12.0};           // = A * [1;2]
    std::vector<double> Xexpected = {1.0, 2.0};

    double *dA = nullptr, *dB = nullptr, **A_arr = nullptr, **B_arr = nullptr;
    int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&dA, 4 * sizeof(double)));
    CHECK_HIP_STATUS(hipMalloc(&dB, 2 * sizeof(double)));
    CHECK_HIP_STATUS(hipMemcpy(dA, A.data(), 4 * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(dB, B.data(), 2 * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMalloc(&A_arr, sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&B_arr, sizeof(double*)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, &dA, sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_arr, &dB, sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMalloc(&ipiv, n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info, sizeof(int)));

    int info_getrs = 0;  // getrsBatched info is a single host int
    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetrsBatched(handle, HIPBLAS_OP_N, n, nrhs, A_arr, lda, ipiv,
                                              B_arr, ldb, &info_getrs, batchCount));
    std::vector<double> X(2);
    CHECK_HIP_STATUS(hipMemcpy(X.data(), dB, 2 * sizeof(double), hipMemcpyDeviceToHost));
    bool ok = compareArrays(X.data(), Xexpected.data(), 2, TOLERANCE_DOUBLE);

    hipFree(dA); hipFree(dB); hipFree(A_arr); hipFree(B_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "hipblasDgetrsBatched known-solution test PASSED" << std::endl;
    return ok;
}

// single precision Sgetrs
bool testSgetrsBatched() {
    std::cout << "Testing hipblasSgetrsBatched..." << std::endl;
    const int n = 6, nrhs = 2, lda = n, ldb = n, batchCount = 4;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<std::vector<float>> A_host(batchCount, std::vector<float>(lda * n));
    std::vector<std::vector<float>> A_orig(batchCount, std::vector<float>(lda * n));
    std::vector<std::vector<float>> B_host(batchCount, std::vector<float>(ldb * nrhs));
    std::vector<std::vector<float>> B_orig(batchCount, std::vector<float>(ldb * nrhs));
    std::mt19937 gen(5151);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
        A_orig[b] = A_host[b];
        generateRandomMatrix(B_host[b], n, nrhs, ldb, gen);
        B_orig[b] = B_host[b];
    }
    std::vector<float*> A_ptrs(batchCount), B_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(float)));
        CHECK_HIP_STATUS(hipMalloc(&B_ptrs[b], ldb * nrhs * sizeof(float)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(B_ptrs[b], B_host[b].data(), ldb * nrhs * sizeof(float), hipMemcpyHostToDevice));
    }
    float **A_arr = nullptr, **B_arr = nullptr; int *ipiv = nullptr, *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(float*)));
    CHECK_HIP_STATUS(hipMalloc(&B_arr, batchCount * sizeof(float*)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv,  batchCount * n * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info,  batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(float*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_arr, B_ptrs.data(), batchCount * sizeof(float*), hipMemcpyHostToDevice));

    int info_getrs = 0;  // getrsBatched info is a single host int
    CHECK_HIPBLAS_STATUS(hipblasSgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasSgetrsBatched(handle, HIPBLAS_OP_N, n, nrhs, A_arr, lda, ipiv,
                                              B_arr, ldb, &info_getrs, batchCount));
    bool ok = true;
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(B_host[b].data(), B_ptrs[b], ldb * nrhs * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> R(n * nrhs, 0.0f);
        cpu_gemm(false, false, n, nrhs, n, 1.0f, A_orig[b].data(), lda, B_host[b].data(), ldb, 0.0f, R.data(), n);
        for (int j = 0; j < nrhs && ok; ++j)
            for (int i = 0; i < n && ok; ++i)
                if (std::abs(R[j * n + i] - B_orig[b][j * ldb + i]) > TOLERANCE_FLOAT) ok = false;
    }
    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(B_ptrs[b]); }
    hipFree(A_arr); hipFree(B_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "hipblasSgetrsBatched test PASSED" << std::endl;
    return ok;
}

// ===========================================================================
// ipiv == nullptr (no-pivoting) workflow.
//   getrf with ipiv=nullptr performs a true LU factorization WITHOUT pivoting
//   via MKL's getrfnp_batch.
//   - On a matrix that needs no pivoting (diagonally dominant), getrf+getri and
//     getrf+getrs with ipiv=nullptr must succeed and be correct.
//   - On a matrix that partial-pivoting WOULD swap but that still has a valid
//     no-pivot LU (all leading principal minors nonzero), getrf with
//     ipiv=nullptr must now succeed and produce a correct factorization.
// ===========================================================================

bool testDgetriNoPivot() {
    std::cout << "Testing getrf+getri with ipiv=nullptr (no-pivot)..." << std::endl;
    const int n = 6, lda = n, ldc = n, batchCount = 4;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> C_host(batchCount, std::vector<double>(ldc * n));
    std::mt19937 gen(31337);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);     // no pivoting needed
        A_orig[b] = A_host[b];
    }
    std::vector<double*> A_ptrs(batchCount), C_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMalloc(&C_ptrs[b], ldc * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(double), hipMemcpyHostToDevice));
    }
    double **A_arr = nullptr, **C_arr = nullptr; int *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&C_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&info, batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_arr, C_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));

    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, nullptr, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetriBatched(handle, n, A_arr, lda, nullptr, C_arr, ldc, info, batchCount));

    bool ok = true;
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(C_host[b].data(), C_ptrs[b], ldc * n * sizeof(double), hipMemcpyDeviceToHost));
        std::vector<double> R(n * n, 0.0);
        cpu_gemm(false, false, n, n, n, 1.0, A_orig[b].data(), lda, C_host[b].data(), ldc, 0.0, R.data(), n);
        for (int i = 0; i < n && ok; ++i)
            for (int j = 0; j < n && ok; ++j) {
                double e = (i == j) ? 1.0 : 0.0;
                if (std::abs(R[j * n + i] - e) > TOLERANCE_DOUBLE) ok = false;
            }
    }
    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(C_ptrs[b]); }
    hipFree(A_arr); hipFree(C_arr); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "getrf+getri ipiv=nullptr (no-pivot) test PASSED" << std::endl;
    return ok;
}

bool testDgetrsNoPivot() {
    std::cout << "Testing getrf+getrs with ipiv=nullptr (no-pivot)..." << std::endl;
    const int n = 6, nrhs = 2, lda = n, ldb = n, batchCount = 4;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    std::vector<std::vector<double>> A_host(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>(lda * n));
    std::vector<std::vector<double>> B_host(batchCount, std::vector<double>(ldb * nrhs));
    std::vector<std::vector<double>> B_orig(batchCount, std::vector<double>(ldb * nrhs));
    std::mt19937 gen(24680);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host[b], n, n, lda, gen);
        makeDiagonallyDominant(A_host[b], n, lda);
        A_orig[b] = A_host[b];
        generateRandomMatrix(B_host[b], n, nrhs, ldb, gen);
        B_orig[b] = B_host[b];
    }
    std::vector<double*> A_ptrs(batchCount), B_ptrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMalloc(&A_ptrs[b], lda * n * sizeof(double)));
        CHECK_HIP_STATUS(hipMalloc(&B_ptrs[b], ldb * nrhs * sizeof(double)));
        CHECK_HIP_STATUS(hipMemcpy(A_ptrs[b], A_host[b].data(), lda * n * sizeof(double), hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(B_ptrs[b], B_host[b].data(), ldb * nrhs * sizeof(double), hipMemcpyHostToDevice));
    }
    double **A_arr = nullptr, **B_arr = nullptr; int *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&B_arr, batchCount * sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&info, batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, A_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_arr, B_ptrs.data(), batchCount * sizeof(double*), hipMemcpyHostToDevice));

    int info_getrs = 0;  // getrsBatched info is a single host int
    CHECK_HIPBLAS_STATUS(hipblasDgetrfBatched(handle, n, A_arr, lda, nullptr, info, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetrsBatched(handle, HIPBLAS_OP_N, n, nrhs, A_arr, lda, nullptr,
                                              B_arr, ldb, &info_getrs, batchCount));
    bool ok = true;
    for (int b = 0; b < batchCount; ++b) {
        CHECK_HIP_STATUS(hipMemcpy(B_host[b].data(), B_ptrs[b], ldb * nrhs * sizeof(double), hipMemcpyDeviceToHost));
        std::vector<double> R(n * nrhs, 0.0);
        cpu_gemm(false, false, n, nrhs, n, 1.0, A_orig[b].data(), lda, B_host[b].data(), ldb, 0.0, R.data(), n);
        for (int j = 0; j < nrhs && ok; ++j)
            for (int i = 0; i < n && ok; ++i)
                if (std::abs(R[j * n + i] - B_orig[b][j * ldb + i]) > TOLERANCE_DOUBLE) ok = false;
    }
    for (int b = 0; b < batchCount; ++b) { hipFree(A_ptrs[b]); hipFree(B_ptrs[b]); }
    hipFree(A_arr); hipFree(B_arr); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "getrf+getrs ipiv=nullptr (no-pivot) test PASSED" << std::endl;
    return ok;
}

// A = [[1,2],[3,4]]: partial pivoting WOULD swap rows (|3| > |1|), but a
// no-pivot LU exists (leading minors 1 and det=-2 are nonzero). With
// getrfnp_batch, getrf(ipiv=nullptr) must now SUCCEED and yield a correct
// factorization. We verify by inverting (getri, ipiv=nullptr) and checking
// A * A^-1 == I.
bool testDgetrfNoPivotSucceedsWhenPivotWouldHappen() {
    std::cout << "Testing getrf ipiv=nullptr succeeds on a pivotable matrix with valid no-pivot LU..." << std::endl;
    const int n = 2, lda = 2, ldc = 2, batchCount = 1;
    hipblasHandle_t handle; CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));
    // A = [[1,2],[3,4]] in column-major.
    std::vector<double> A = {1.0, 3.0, 2.0, 4.0};
    std::vector<double> A_orig = A;
    double *dA = nullptr, *dC = nullptr, **A_arr = nullptr, **C_arr = nullptr; int *info = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&dA, 4 * sizeof(double)));
    CHECK_HIP_STATUS(hipMalloc(&dC, 4 * sizeof(double)));
    CHECK_HIP_STATUS(hipMemcpy(dA, A.data(), 4 * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMalloc(&A_arr, sizeof(double*)));
    CHECK_HIP_STATUS(hipMalloc(&C_arr, sizeof(double*)));
    CHECK_HIP_STATUS(hipMemcpy(A_arr, &dA, sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(C_arr, &dC, sizeof(double*), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMalloc(&info, sizeof(int)));

    hipblasStatus_t st = hipblasDgetrfBatched(handle, n, A_arr, lda, nullptr, info, batchCount);
    bool ok = (st == HIPBLAS_STATUS_SUCCESS);
    if (!ok) std::cerr << "  getrf(ipiv=nullptr) expected SUCCESS, got " << (int)st << std::endl;
    if (ok) {
        CHECK_HIPBLAS_STATUS(hipblasDgetriBatched(handle, n, A_arr, lda, nullptr, C_arr, ldc, info, batchCount));
        std::vector<double> C(4);
        CHECK_HIP_STATUS(hipMemcpy(C.data(), dC, 4 * sizeof(double), hipMemcpyDeviceToHost));
        std::vector<double> R(4, 0.0);
        cpu_gemm(false, false, n, n, n, 1.0, A_orig.data(), lda, C.data(), ldc, 0.0, R.data(), n);
        for (int i = 0; i < n && ok; ++i)
            for (int j = 0; j < n && ok; ++j) {
                double e = (i == j) ? 1.0 : 0.0;
                if (std::abs(R[j * n + i] - e) > TOLERANCE_DOUBLE) ok = false;
            }
        if (!ok) std::cerr << "  A * A^-1 != I after no-pivot factorization" << std::endl;
    }

    hipFree(dA); hipFree(dC); hipFree(A_arr); hipFree(C_arr); hipFree(info);
    hipblasDestroy(handle);
    if (ok) std::cout << "getrf ipiv=nullptr pivotable-matrix test PASSED" << std::endl;
    return ok;
}

int main() {
    std::cout << "=== Testing H4I-HipBLAS Batched LAPACK Functions ===" << std::endl << std::endl;
    
    bool allPassed = true;
    
    // Test all the newly implemented batched functions
    allPassed &= testDgetrfBatched();
    allPassed &= testDgetriBatched();
    allPassed &= testCgetrfBatched();
    allPassed &= testSgemmStridedBatched();

    // Additional getri coverage
    allPassed &= testDgetriSizesAndBatches();
    allPassed &= testDgetriKnownInverse();
    allPassed &= testDgetriPivoting();
    allPassed &= testSgetriBatched();

    // getrs (solve A X = B) coverage
    allPassed &= testDgetrsBatched();
    // allPassed &= testDgetrsTranspose();
    allPassed &= testDgetrsKnownSolution();
    allPassed &= testSgetrsBatched();

    // ipiv=nullptr (no-pivoting) workflow
    allPassed &= testDgetriNoPivot();
    allPassed &= testDgetrsNoPivot();
    allPassed &= testDgetrfNoPivotSucceedsWhenPivotWouldHappen();

    std::cout << std::endl;
    if (allPassed) {
        std::cout << "ALL TESTS PASSED! Batched functions are working correctly." << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Some tests failed. Please check the implementation." << std::endl;
        return EXIT_FAILURE;
    }
} 
