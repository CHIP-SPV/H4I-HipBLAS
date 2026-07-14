// Correctness tests for the StridedBatched LAPACK entry points
//   hipblas{S,D}getrfStridedBatched
//   hipblas{S,D}getrsStridedBatched
//
// Coverage strategy:
//   1. Math checks that do not depend on any other hipBLAS path:
//        getrf : verify P*A == L*U per batch
//        getrs : verify op(A_orig) * X == B_orig per batch
//   2. Cross-check against the *batched* (pointer-array) path on identical data:
//        run getrfStridedBatched and getrfBatched on the same matrices and require
//        bit-for-similar agreement (LU factors + pivots). This is the "extra layer"
//        the batched tests give us for free.
//   3. Strided-specific coverage the batched path cannot express:
//        padded strides (strideA/strideB/strideP strictly larger than the minimal
//        contiguous stride), so a stride bug that a contiguous layout would hide
//        gets caught.
//   4. no-pivot (ipiv == nullptr) workflow.
//
// The helpers (compareArrays, generateRandomMatrix, makeDiagonallyDominant,
// cpu_gemm, applyRowPermutations) mirror those in batched_correctness_test.cpp.

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "hipblas.h"

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
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            std::cerr << "Array element at index " << i << ": " << a[i] << " or " << b[i]
                      << " is nan. This should never happen." << std::endl;
            return false;
        }
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                      << " (diff: " << std::abs(a[i] - b[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// Column-major random matrix into the leading rows x cols block of an lda-tall column.
template<typename T>
void generateRandomMatrix(std::vector<T>& matrix, int rows, int cols, int lda,
                          size_t offset, std::mt19937& gen) {
    std::uniform_real_distribution<typename std::conditional<std::is_same<T, float>::value,
                                                             float, double>::type> dis(-1.0, 1.0);
    for (int col = 0; col < cols; col++)
        for (int row = 0; row < rows; row++)
            matrix[offset + col * lda + row] = static_cast<T>(dis(gen));
}

template<typename T>
void makeDiagonallyDominant(std::vector<T>& matrix, int n, int lda, size_t offset) {
    for (int i = 0; i < n; i++) {
        T sum = 0;
        for (int j = 0; j < n; j++)
            if (i != j) sum += std::abs(matrix[offset + j * lda + i]);
        matrix[offset + i * lda + i] = sum + static_cast<T>(1.0);
    }
}

// C = alpha*op(A)*op(B) + beta*C, all column-major (matches batched test reference).
template<typename T>
void cpu_gemm(bool transA, bool transB, int m, int n, int k,
              T alpha, const T* A, int lda, const T* B, int ldb,
              T beta, T* C, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            T sum = static_cast<T>(0);
            for (int l = 0; l < k; l++) {
                T a_val = transA ? A[i * lda + l] : A[l * lda + i];
                T b_val = transB ? B[l * ldb + j] : B[j * ldb + l];
                sum += a_val * b_val;
            }
            C[j * ldc + i] = alpha * sum + beta * C[j * ldc + i];
        }
    }
}

// Apply LAPACK 1-based row interchanges to a column-major n x n matrix block.
template<typename T>
void applyRowPermutations(std::vector<T>& mat, const int* piv, int n, int lda, size_t offset) {
    for (int i = 0; i < n; ++i) {
        int jp = piv[i] - 1;
        if (jp != i)
            for (int col = 0; col < n; ++col)
                std::swap(mat[offset + col * lda + i], mat[offset + col * lda + jp]);
    }
}

// ===========================================================================
// getrf strided: verify P*A == L*U, and cross-check factors/pivots vs the
// batched path on identical inputs. strideA / strideP may be padded.
// ===========================================================================
template<typename T>
static bool runGetrfStridedCorrectness(int n, int batchCount, int ldaPad, int stridePadA,
                                       int stridePadP, unsigned seed, T tol,
                                       hipblasStatus_t (*getrfStrided)(hipblasHandle_t, int, T*, int,
                                                                       hipblasStride, int*, hipblasStride,
                                                                       int*, int),
                                       hipblasStatus_t (*getrfBatched)(hipblasHandle_t, int, T* const[],
                                                                       int, int*, int*, int),
                                       const char* tag) {
    const int lda      = n + ldaPad;               // padded leading dim
    const int64_t sA   = (int64_t)lda * n + stridePadA;  // padded matrix stride
    const int64_t sP   = (int64_t)n + stridePadP;        // padded pivot stride

    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    // Host: one big strided A buffer, plus the pristine originals for P*A==L*U.
    std::vector<T> A_host((size_t)sA * batchCount, T(0));
    std::vector<std::vector<T>> A_orig(batchCount, std::vector<T>((size_t)lda * n, T(0)));
    std::mt19937 gen(seed);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host, n, n, lda, (size_t)b * sA, gen);
        makeDiagonallyDominant(A_host, n, lda, (size_t)b * sA);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A_orig[b][j * lda + i] = A_host[(size_t)b * sA + j * lda + i];
    }

    // --- strided path ---
    T*   A_dev = nullptr;
    int* ipiv_dev = nullptr;
    int* info_dev = nullptr;
    CHECK_HIP_STATUS(hipMalloc(&A_dev,    A_host.size() * sizeof(T)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv_dev, (size_t)sP * batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info_dev, batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(T), hipMemcpyHostToDevice));

    CHECK_HIPBLAS_STATUS(getrfStrided(handle, n, A_dev, lda, sA, ipiv_dev, sP, info_dev, batchCount));

    std::vector<T>   LU_host(A_host.size());
    std::vector<int> ipiv_host((size_t)sP * batchCount);
    std::vector<int> info_host(batchCount);
    CHECK_HIP_STATUS(hipMemcpy(LU_host.data(), A_dev, A_host.size() * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_STATUS(hipMemcpy(ipiv_host.data(), ipiv_dev, (size_t)sP * batchCount * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP_STATUS(hipMemcpy(info_host.data(), info_dev, batchCount * sizeof(int), hipMemcpyDeviceToHost));

    bool ok = true;
    for (int b = 0; b < batchCount && ok; ++b) {
        if (info_host[b] != 0) {
            std::cerr << tag << ": info[" << b << "] = " << info_host[b] << std::endl;
            ok = false;
            break;
        }
        // Build L, U from the strided LU block for batch b.
        std::vector<T> L(n * n, T(0)), U(n * n, T(0));
        const size_t off = (size_t)b * sA;
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row) {
                T val = LU_host[off + col * lda + row];
                if (row > col)       L[col * n + row] = val;
                else if (row == col) { L[col * n + row] = T(1); U[col * n + row] = val; }
                else                 U[col * n + row] = val;
            }
        // P*A_orig (apply pivots read from the padded-stride pivot buffer).
        std::vector<T> P_A = A_orig[b];
        applyRowPermutations(P_A, &ipiv_host[(size_t)b * sP], n, lda, 0);
        // Compare against L*U.
        std::vector<T> LU_recon(n * n, T(0));
        cpu_gemm<T>(false, false, n, n, n, T(1), L.data(), n, U.data(), n, T(0), LU_recon.data(), n);
        // P_A is lda-tall; compress to n x n for the comparison.
        std::vector<T> P_A_nn(n * n);
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row)
                P_A_nn[col * n + row] = P_A[col * lda + row];
        if (!compareArrays(P_A_nn.data(), LU_recon.data(), n * n, tol)) {
            std::cerr << tag << ": P*A != L*U for batch " << b << std::endl;
            ok = false;
        }
    }

    // --- cross-check vs batched path on identical inputs ---
    if (ok) {
        std::vector<T*> Ab_ptrs(batchCount);
        for (int b = 0; b < batchCount; ++b) {
            CHECK_HIP_STATUS(hipMalloc(&Ab_ptrs[b], (size_t)lda * n * sizeof(T)));
            // copy the same (contiguous n columns of the) original into a compact lda*n block
            std::vector<T> compact((size_t)lda * n, T(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    compact[j * lda + i] = A_orig[b][j * lda + i];
            CHECK_HIP_STATUS(hipMemcpy(Ab_ptrs[b], compact.data(), (size_t)lda * n * sizeof(T), hipMemcpyHostToDevice));
        }
        T** A_arr = nullptr; int* ipiv_b = nullptr; int* info_b = nullptr;
        CHECK_HIP_STATUS(hipMalloc(&A_arr, batchCount * sizeof(T*)));
        CHECK_HIP_STATUS(hipMalloc(&ipiv_b, (size_t)batchCount * n * sizeof(int)));
        CHECK_HIP_STATUS(hipMalloc(&info_b, batchCount * sizeof(int)));
        CHECK_HIP_STATUS(hipMemcpy(A_arr, Ab_ptrs.data(), batchCount * sizeof(T*), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_STATUS(getrfBatched(handle, n, A_arr, lda, ipiv_b, info_b, batchCount));

        std::vector<int> ipiv_bh((size_t)batchCount * n);
        CHECK_HIP_STATUS(hipMemcpy(ipiv_bh.data(), ipiv_b, (size_t)batchCount * n * sizeof(int), hipMemcpyDeviceToHost));
        for (int b = 0; b < batchCount && ok; ++b) {
            std::vector<T> LUb((size_t)lda * n);
            CHECK_HIP_STATUS(hipMemcpy(LUb.data(), Ab_ptrs[b], (size_t)lda * n * sizeof(T), hipMemcpyDeviceToHost));
            // Compare LU factors (leading n x n) and pivots against the strided result.
            for (int j = 0; j < n && ok; ++j)
                for (int i = 0; i < n && ok; ++i)
                    if (std::abs(LUb[j * lda + i] - LU_host[(size_t)b * sA + j * lda + i]) > tol) {
                        std::cerr << tag << ": strided vs batched LU mismatch batch " << b
                                  << " (" << i << "," << j << ")" << std::endl;
                        ok = false;
                    }
            for (int i = 0; i < n && ok; ++i)
                if (ipiv_bh[b * n + i] != ipiv_host[(size_t)b * sP + i]) {
                    std::cerr << tag << ": strided vs batched pivot mismatch batch " << b
                              << " idx " << i << std::endl;
                    ok = false;
                }
        }
        for (int b = 0; b < batchCount; ++b) hipFree(Ab_ptrs[b]);
        hipFree(A_arr); hipFree(ipiv_b); hipFree(info_b);
    }

    hipFree(A_dev); hipFree(ipiv_dev); hipFree(info_dev);
    hipblasDestroy(handle);
    return ok;
}

// ===========================================================================
// getrs strided: factor with getrfStrided, solve with getrsStrided, verify
// op(A_orig) * X == B_orig. Padded strides on A, B and pivots.
// ===========================================================================
template<typename T>
static bool runGetrsStridedCorrectness(hipblasOperation_t trans, int n, int nrhs, int batchCount,
                                       int ldaPad, int ldbPad, int stridePadA, int stridePadB,
                                       int stridePadP, unsigned seed, T tol,
                                       hipblasStatus_t (*getrfStrided)(hipblasHandle_t, int, T*, int,
                                                                       hipblasStride, int*, hipblasStride,
                                                                       int*, int),
                                       hipblasStatus_t (*getrsStrided)(hipblasHandle_t, hipblasOperation_t,
                                                                       int, int, T*, int, hipblasStride,
                                                                       const int*, hipblasStride, T*, int,
                                                                       hipblasStride, int*, int),
                                       const char* tag) {
    const int lda    = n + ldaPad;
    const int ldb    = n + ldbPad;
    const int64_t sA = (int64_t)lda * n + stridePadA;
    const int64_t sB = (int64_t)ldb * nrhs + stridePadB;
    const int64_t sP = (int64_t)n + stridePadP;

    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<T> A_host((size_t)sA * batchCount, T(0));
    std::vector<T> B_host((size_t)sB * batchCount, T(0));
    std::vector<std::vector<T>> A_orig(batchCount, std::vector<T>((size_t)lda * n, T(0)));
    std::vector<std::vector<T>> B_orig(batchCount, std::vector<T>((size_t)ldb * nrhs, T(0)));
    std::mt19937 gen(seed);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host, n, n, lda, (size_t)b * sA, gen);
        makeDiagonallyDominant(A_host, n, lda, (size_t)b * sA);
        generateRandomMatrix(B_host, n, nrhs, ldb, (size_t)b * sB, gen);
        for (int j = 0; j < n;    ++j) for (int i = 0; i < n; ++i) A_orig[b][j*lda+i] = A_host[(size_t)b*sA + j*lda + i];
        for (int j = 0; j < nrhs; ++j) for (int i = 0; i < n; ++i) B_orig[b][j*ldb+i] = B_host[(size_t)b*sB + j*ldb + i];
    }

    T*   A_dev = nullptr;
    T*   B_dev = nullptr;
    int* ipiv_dev = nullptr;
    int* info_dev = nullptr;
    int  info_getrs = 0;  // getrs info is a single host int (matches batched path)
    CHECK_HIP_STATUS(hipMalloc(&A_dev,    A_host.size() * sizeof(T)));
    CHECK_HIP_STATUS(hipMalloc(&B_dev,    B_host.size() * sizeof(T)));
    CHECK_HIP_STATUS(hipMalloc(&ipiv_dev, (size_t)sP * batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMalloc(&info_dev, batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_dev, B_host.data(), B_host.size() * sizeof(T), hipMemcpyHostToDevice));

    CHECK_HIPBLAS_STATUS(getrfStrided(handle, n, A_dev, lda, sA, ipiv_dev, sP, info_dev, batchCount));
    CHECK_HIPBLAS_STATUS(getrsStrided(handle, trans, n, nrhs, A_dev, lda, sA, ipiv_dev, sP,
                                      B_dev, ldb, sB, &info_getrs, batchCount));

    CHECK_HIP_STATUS(hipMemcpy(B_host.data(), B_dev, B_host.size() * sizeof(T), hipMemcpyDeviceToHost));

    bool ok = true;
    const bool tr = (trans != HIPBLAS_OP_N);
    for (int b = 0; b < batchCount && ok; ++b) {
        // X is in B_host[b] block (leading n x nrhs). R = op(A_orig)*X, compare to B_orig.
        std::vector<T> X(n * nrhs);
        for (int j = 0; j < nrhs; ++j) for (int i = 0; i < n; ++i) X[j*n+i] = B_host[(size_t)b*sB + j*ldb + i];
        std::vector<T> R(n * nrhs, T(0));
        cpu_gemm<T>(tr, false, n, nrhs, n, T(1), A_orig[b].data(), lda, X.data(), n, T(0), R.data(), n);
        for (int j = 0; j < nrhs && ok; ++j)
            for (int i = 0; i < n && ok; ++i)
                if (std::abs(R[j*n+i] - B_orig[b][j*ldb+i]) > tol) {
                    std::cerr << tag << ": op(A)*X != B for batch " << b
                              << " (" << i << "," << j << ")" << std::endl;
                    ok = false;
                }
    }

    hipFree(A_dev); hipFree(B_dev); hipFree(ipiv_dev); hipFree(info_dev);
    hipblasDestroy(handle);
    return ok;
}

// ---------------------------------------------------------------------------
// getrs strided, no-pivot solve (ipiv == nullptr). A must be factored with the
// no-pivot getrf (ipiv == nullptr) so the LU has no row swaps.
// ---------------------------------------------------------------------------
static bool testDgetrsStridedNoPivot() {
    std::cout << "Testing hipblasDgetrsStridedBatched no-pivot (ipiv=nullptr)..." << std::endl;
    const int n = 6, nrhs = 2, batchCount = 4;
    const int lda = n, ldb = n;
    const int64_t sA = (int64_t)lda * n, sB = (int64_t)ldb * nrhs;

    hipblasHandle_t handle;
    CHECK_HIPBLAS_STATUS(hipblasCreate(&handle));

    std::vector<double> A_host((size_t)sA * batchCount, 0.0);
    std::vector<double> B_host((size_t)sB * batchCount, 0.0);
    std::vector<std::vector<double>> A_orig(batchCount, std::vector<double>((size_t)lda * n));
    std::vector<std::vector<double>> B_orig(batchCount, std::vector<double>((size_t)ldb * nrhs));
    std::mt19937 gen(4242);
    for (int b = 0; b < batchCount; ++b) {
        generateRandomMatrix(A_host, n, n, lda, (size_t)b * sA, gen);
        makeDiagonallyDominant(A_host, n, lda, (size_t)b * sA);  // no-pivot safe
        generateRandomMatrix(B_host, n, nrhs, ldb, (size_t)b * sB, gen);
        for (int j = 0; j < n;    ++j) for (int i = 0; i < n; ++i) A_orig[b][j*lda+i] = A_host[(size_t)b*sA + j*lda + i];
        for (int j = 0; j < nrhs; ++j) for (int i = 0; i < n; ++i) B_orig[b][j*ldb+i] = B_host[(size_t)b*sB + j*ldb + i];
    }

    double *A_dev = nullptr, *B_dev = nullptr; int* info_dev = nullptr; int info_getrs = 0;
    CHECK_HIP_STATUS(hipMalloc(&A_dev, A_host.size() * sizeof(double)));
    CHECK_HIP_STATUS(hipMalloc(&B_dev, B_host.size() * sizeof(double)));
    CHECK_HIP_STATUS(hipMalloc(&info_dev, batchCount * sizeof(int)));
    CHECK_HIP_STATUS(hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(B_dev, B_host.data(), B_host.size() * sizeof(double), hipMemcpyHostToDevice));

    // Factor with no pivoting, then solve with no pivoting (ipiv = nullptr, strideP = 0).
    CHECK_HIPBLAS_STATUS(hipblasDgetrfStridedBatched(handle, n, A_dev, lda, sA, nullptr, 0, info_dev, batchCount));
    CHECK_HIPBLAS_STATUS(hipblasDgetrsStridedBatched(handle, HIPBLAS_OP_N, n, nrhs, A_dev, lda, sA,
                                                     nullptr, 0, B_dev, ldb, sB, &info_getrs, batchCount));

    CHECK_HIP_STATUS(hipMemcpy(B_host.data(), B_dev, B_host.size() * sizeof(double), hipMemcpyDeviceToHost));

    bool ok = true;
    for (int b = 0; b < batchCount && ok; ++b) {
        std::vector<double> X(n * nrhs);
        for (int j = 0; j < nrhs; ++j) for (int i = 0; i < n; ++i) X[j*n+i] = B_host[(size_t)b*sB + j*ldb + i];
        std::vector<double> R(n * nrhs, 0.0);
        cpu_gemm<double>(false, false, n, nrhs, n, 1.0, A_orig[b].data(), lda, X.data(), n, 0.0, R.data(), n);
        for (int j = 0; j < nrhs && ok; ++j)
            for (int i = 0; i < n && ok; ++i)
                if (std::abs(R[j*n+i] - B_orig[b][j*ldb+i]) > TOLERANCE_DOUBLE) ok = false;
    }

    hipFree(A_dev); hipFree(B_dev); hipFree(info_dev);
    hipblasDestroy(handle);
    if (ok) std::cout << "  no-pivot strided getrs PASSED" << std::endl;
    else    std::cerr << "  no-pivot strided getrs FAILED" << std::endl;
    return ok;
}

// ---------------------------------------------------------------------------
// Drivers that sweep sizes / batch counts / padded strides.
// ---------------------------------------------------------------------------
static bool testDgetrfStrided() {
    std::cout << "Testing hipblasDgetrfStridedBatched (P*A==L*U, cross-check vs batched, padded strides)..." << std::endl;
    const int sizes[]   = {1, 3, 8, 16};
    const int batches[] = {1, 5};
    // (ldaPad, stridePadA, stridePadP): last row exercises padding beyond minimal.
    const int pads[][3] = {{0,0,0}, {2,3,1}};
    bool ok = true;
    for (int n : sizes)
      for (int bc : batches)
        for (auto& p : pads) {
            bool r = runGetrfStridedCorrectness<double>(n, bc, p[0], p[1], p[2],
                        100u + 7u*(unsigned)n + (unsigned)bc + (unsigned)p[0],
                        TOLERANCE_DOUBLE, hipblasDgetrfStridedBatched, hipblasDgetrfBatched, "Dgetrf strided");
            std::cout << "  n=" << n << " batch=" << bc << " pad(lda,sA,sP)=(" << p[0] << "," << p[1]
                      << "," << p[2] << ") : " << (r ? "ok" : "FAIL") << std::endl;
            ok = ok && r;
        }
    if (ok) std::cout << "hipblasDgetrfStridedBatched test PASSED" << std::endl;
    return ok;
}

static bool testSgetrfStrided() {
    std::cout << "Testing hipblasSgetrfStridedBatched (single precision)..." << std::endl;
    bool ok = runGetrfStridedCorrectness<float>(10, 4, 1, 2, 1, 555u, TOLERANCE_FLOAT,
                    hipblasSgetrfStridedBatched, hipblasSgetrfBatched, "Sgetrf strided");
    if (ok) std::cout << "hipblasSgetrfStridedBatched test PASSED" << std::endl;
    return ok;
}

static bool testDgetrsStrided() {
    std::cout << "Testing hipblasDgetrsStridedBatched (op(A)*X==B, padded strides, N and T)..." << std::endl;
    const int sizes[] = {1, 4, 9, 16};
    const int nrhss[] = {1, 3};
    const int batches[] = {1, 6};
    bool ok = true;
    for (int n : sizes)
      for (int nrhs : nrhss)
        for (int bc : batches) {
            bool r = runGetrsStridedCorrectness<double>(HIPBLAS_OP_N, n, nrhs, bc,
                        1, 2, 3, 2, 1, 300u + 11u*(unsigned)n + 3u*(unsigned)nrhs + (unsigned)bc,
                        TOLERANCE_DOUBLE, hipblasDgetrfStridedBatched, hipblasDgetrsStridedBatched, "Dgetrs strided N");
            std::cout << "  n=" << n << " nrhs=" << nrhs << " batch=" << bc
                      << " : " << (r ? "ok" : "FAIL") << std::endl;
            ok = ok && r;
        }
    // transpose solve
    bool rt = runGetrsStridedCorrectness<double>(HIPBLAS_OP_T, 9, 3, 4, 1, 1, 2, 2, 1, 9191u,
                    TOLERANCE_DOUBLE, hipblasDgetrfStridedBatched, hipblasDgetrsStridedBatched, "Dgetrs strided T");
    std::cout << "  transpose (A^T X = B) n=9 nrhs=3 batch=4 : " << (rt ? "ok" : "FAIL") << std::endl;
    ok = ok && rt;
    if (ok) std::cout << "hipblasDgetrsStridedBatched test PASSED" << std::endl;
    return ok;
}

static bool testSgetrsStrided() {
    std::cout << "Testing hipblasSgetrsStridedBatched (single precision)..." << std::endl;
    bool ok = runGetrsStridedCorrectness<float>(HIPBLAS_OP_N, 12, 2, 5, 1, 1, 2, 2, 1, 777u,
                    TOLERANCE_FLOAT, hipblasSgetrfStridedBatched, hipblasSgetrsStridedBatched, "Sgetrs strided");
    if (ok) std::cout << "hipblasSgetrsStridedBatched test PASSED" << std::endl;
    return ok;
}

int main() {
    std::cout << "=== Testing H4I-HipBLAS StridedBatched LAPACK Functions ===" << std::endl << std::endl;
    bool allPassed = true;

    allPassed &= testDgetrfStrided();
    allPassed &= testSgetrfStrided();
    allPassed &= testDgetrsStrided();
    allPassed &= testSgetrsStrided();
    allPassed &= testDgetrsStridedNoPivot();

    std::cout << std::endl;
    if (allPassed) {
        std::cout << "ALL TESTS PASSED! StridedBatched functions are working correctly." << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Some tests failed. Please check the implementation." << std::endl;
        return EXIT_FAILURE;
    }
}
