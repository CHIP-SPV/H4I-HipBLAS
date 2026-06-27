// Minimal HIP + hipBLAS: solve A^T X = B for one 2x2 system.
//   A = [[4,3],[6,3]] col-major {4,6,3,3};  A^T=[[4,6],[3,3]];  B=[16;9];  X must be [1;2].
// Same problem as two_sycl.cpp, through the hipBLAS -> MKLShim -> oneMKL path.
//
// Build: hipcc two_hip.cpp -I<build>/include -L<build>/src -lhipblas -o two_hip
// Run:   CHIP_BE=level0 ./two_hip

#include <cstdio>
#include <hip/hip_runtime.h>
#include "hipblas.h"

int main() {
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    const int n = 2, nrhs = 1, lda = 2, ldb = 2, batchCount = 1;
    double A[4] = {4, 6, 3, 3};   // col-major A = [[4,3],[6,3]]
    double B[2] = {16, 9};        // = A^T * [1;2]

    std::printf("A (col-major) = {%.4g, %.4g, %.4g, %.4g}\n", A[0], A[1], A[2], A[3]);
    std::printf("  A = [ %.4g  %.4g ]\n      [ %.4g  %.4g ]\n", A[0], A[2], A[1], A[3]);
    std::printf("B (rhs) = [ %.4g ; %.4g ]   solving A^T X = B\n", B[0], B[1]);

    double *dA = nullptr, *dB = nullptr, **A_arr = nullptr, **B_arr = nullptr;
    int *ipiv = nullptr, *info = nullptr;
    hipMalloc(&dA, 4 * sizeof(double));
    hipMalloc(&dB, 2 * sizeof(double));
    hipMemcpy(dA, A, 4 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dB, B, 2 * sizeof(double), hipMemcpyHostToDevice);
    hipMalloc(&A_arr, sizeof(double*));
    hipMalloc(&B_arr, sizeof(double*));
    hipMemcpy(A_arr, &dA, sizeof(double*), hipMemcpyHostToDevice);
    hipMemcpy(B_arr, &dB, sizeof(double*), hipMemcpyHostToDevice);
    hipMalloc(&ipiv, n * sizeof(int));
    hipMalloc(&info, sizeof(int));

    int info_getrs = 0;  // getrsBatched info is a single host int
    hipblasDgetrfBatched(handle, n, A_arr, lda, ipiv, info, batchCount);
    hipblasDgetrsBatched(handle, HIPBLAS_OP_T, n, nrhs, A_arr, lda, ipiv,
                         B_arr, ldb, &info_getrs, batchCount);

    double X[2] = {0, 0};
    hipMemcpy(X, dB, 2 * sizeof(double), hipMemcpyDeviceToHost);
    std::printf("solution X = [ %.10g ; %.10g ]   expected [ 1 ; 2 ]\n", X[0], X[1]);

    hipFree(dA); hipFree(dB); hipFree(A_arr); hipFree(B_arr); hipFree(ipiv); hipFree(info);
    hipblasDestroy(handle);
    return 0;
}
