// Pure oneMKL reproducer for the batched transpose getrs path.
//
// No chipStar, no hipBLAS, no MKLShim. Just a SYCL queue + oneMKL group-API
// getrf_batch -> getrs_batch(trans=T), mirroring exactly what the shim does
// (device USM pointer arrays filled by memcpy, group API, in-order queue).
//
// Two matched cases, each with a hipBLAS twin that uses IDENTICAL inputs:
//   * n=2 known answer  <-> testDgetrsKnownSolutionTranspose       (hardcoded 2x2)
//   * n=8 known answer  <-> testDgetrsKnownSolutionTransposeLarge  (buildKnownTransposeSystem)
// No RNG: the n=8 system is built by the closed-form buildKnownTransposeSystem()
// below, which MUST stay byte-identical to the copy in batched_correctness_test.cpp.
// If pure oneMKL and the hipBLAS twin ever disagree on the same inputs, the
// difference is in our stack, not oneMKL.
//
// Build (oneAPI loaded):  icpx -fsycl -qmkl mkl_getrs_transpose_repro.cpp -o repro
// Run:                    ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./repro

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// --- Deterministic known-answer transpose system (NO RNG) -------------------
// Build A (n x n, col-major), known X (n x nrhs, col-major), and B = A^T * X.
// Solving A^T X' = B must recover X' == X.
// MUST stay byte-identical to the copy in batched_correctness_test.cpp.
static void buildKnownTransposeSystem(int n, int nrhs,
        std::vector<double>& A, std::vector<double>& X, std::vector<double>& B) {
    const int lda = n, ldb = n;
    A.assign((size_t)n * n, 0.0);
    X.assign((size_t)n * nrhs, 0.0);
    B.assign((size_t)n * nrhs, 0.0);
    // bounded, deterministic entries
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row)
            A[col * lda + row] = 0.5 + 0.1 * (double)(((row * 7 + col * 3) % 5));
    // diagonal dominance -> well-conditioned, non-singular
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) if (i != j) s += std::abs(A[j * lda + i]);
        A[i * lda + i] = s + 1.0;
    }
    // known solution
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[j * ldb + i] = 1.0 + (double)i + 0.25 * (double)j;
    // B = A^T * X ; (A^T)(i,k) = A(k,i) = A[i*lda + k]
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) {
            double r = 0.0;
            for (int k = 0; k < n; ++k) r += A[i * lda + k] * X[j * ldb + k];
            B[j * ldb + i] = r;
        }
}

// Run one batched (group, single matrix) transpose solve on the given A/B,
// return max|X_solved - Xexpected| (NaN-safe -> reported as +inf).
static double run_known(sycl::queue& q, int n, int nrhs,
                        const std::vector<double>& A_host,
                        const std::vector<double>& B_host,
                        const std::vector<double>& Xexp) {
    const int64_t lda = n, ldb = n;
    const int64_t group_count = 1, group_size = 1;
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;

    double*  dA    = sycl::malloc_device<double>((size_t)n * n, q);
    double*  dB    = sycl::malloc_device<double>((size_t)n * nrhs, q);
    int64_t* dIpiv = sycl::malloc_device<int64_t>(n, q);
    q.memcpy(dA, A_host.data(), (size_t)n * n * sizeof(double));
    q.memcpy(dB, B_host.data(), (size_t)n * nrhs * sizeof(double));
    q.wait();

    double**  A_ptrs    = sycl::malloc_device<double*>(group_size, q);
    double**  B_ptrs    = sycl::malloc_device<double*>(group_size, q);
    int64_t** ipiv_ptrs = sycl::malloc_device<int64_t*>(group_size, q);
    double*  hA[1]  = {dA};
    double*  hB[1]  = {dB};
    int64_t* hIp[1] = {dIpiv};
    q.memcpy(A_ptrs,    hA,  sizeof(double*));
    q.memcpy(B_ptrs,    hB,  sizeof(double*));
    q.memcpy(ipiv_ptrs, hIp, sizeof(int64_t*));
    q.wait();

    int64_t m_arr[1]    = {n};
    int64_t n_arr[1]    = {n};
    int64_t nrhs_arr[1] = {nrhs};
    int64_t lda_arr[1]  = {lda};
    int64_t ldb_arr[1]  = {ldb};
    int64_t gsz[1]      = {group_size};
    oneapi::mkl::transpose trans_arr[1] = {trans};

    try {
        int64_t f_scr = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(
            q, m_arr, n_arr, lda_arr, group_count, gsz);
        double* f_scratch = sycl::malloc_device<double>(f_scr > 0 ? f_scr : 1, q);
        oneapi::mkl::lapack::getrf_batch(q, m_arr, n_arr, A_ptrs, lda_arr, ipiv_ptrs,
                                         group_count, gsz, f_scratch, f_scr, {});
        q.wait();

        int64_t s_scr = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(
            q, trans_arr, n_arr, nrhs_arr, lda_arr, ldb_arr, group_count, gsz);
        double* s_scratch = sycl::malloc_device<double>(s_scr > 0 ? s_scr : 1, q);
        oneapi::mkl::lapack::getrs_batch(
            q, trans_arr, n_arr, nrhs_arr,
            const_cast<const double* const*>(A_ptrs), lda_arr,
            const_cast<const int64_t* const*>(ipiv_ptrs),
            B_ptrs, ldb_arr, group_count, gsz, s_scratch, s_scr, {});
        q.wait();

        sycl::free(f_scratch, q);
        sycl::free(s_scratch, q);
    } catch (oneapi::mkl::lapack::exception const& e) {
        std::printf("    MKL LAPACK exception: info=%ld detail=%ld what=%s\n",
                    (long)e.info(), (long)e.detail(), e.what());
    } catch (sycl::exception const& e) {
        std::printf("    SYCL exception: %s\n", e.what());
    }

    std::vector<double> X((size_t)n * nrhs, 0.0);
    q.memcpy(X.data(), dB, (size_t)n * nrhs * sizeof(double)).wait();

    double worst = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double d = std::abs(X[i] - Xexp[i]);
        if (!(d <= worst)) worst = std::isnan(d) ? INFINITY : d;  // NaN-safe
    }

    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dIpiv, q);
    sycl::free(A_ptrs, q); sycl::free(B_ptrs, q); sycl::free(ipiv_ptrs, q);
    return worst;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    MKLVersion v; mkl_get_version(&v);
    std::printf("oneMKL %d.%d update %d (build %s)\n",
                v.MajorVersion, v.MinorVersion, v.UpdateVersion, v.Build);

    const double TOL = 1e-8;
    bool all_ok = true;

    // ---- case 1: n=2 known answer (identical to testDgetrsKnownSolutionTranspose) ----
    {
        // A = [[4,3],[6,3]] col-major {4,6,3,3}; A^T=[[4,6],[3,3]]; X=[1;2]; B=A^T X=[16;9].
        std::vector<double> A = {4.0, 6.0, 3.0, 3.0};
        std::vector<double> B = {16.0, 9.0};
        std::vector<double> Xexp = {1.0, 2.0};
        double worst = run_known(q, 2, 1, A, B, Xexp);
        bool ok = (worst <= TOL); all_ok = all_ok && ok;
        std::printf("  n=2 trans=T (known 2x2)   : max|X-exp| = %.3e  -> %s\n",
                    worst, ok ? "PASS" : "FAIL");
    }

    // ---- case 2: n=8 known answer (identical to testDgetrsKnownSolutionTransposeLarge) ----
    {
        const int n = 8, nrhs = 3;
        std::vector<double> A, X, B;
        buildKnownTransposeSystem(n, nrhs, A, X, B);
        double worst = run_known(q, n, nrhs, A, B, X);
        bool ok = (worst <= TOL); all_ok = all_ok && ok;
        std::printf("  n=8 trans=T (known build) : max|X-exp| = %.3e  -> %s\n",
                    worst, ok ? "PASS" : "FAIL");
    }

    std::printf("\nPure-oneMKL probe: %s\n", all_ok ? "all PASS" : "FAIL (see per-case above)");
    return all_ok ? 0 : 1;
}
