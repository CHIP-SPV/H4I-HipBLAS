// Minimal pure SYCL + oneMKL: solve A^T X = B for one 2x2 system.
//   A = [[4,3],[6,3]] col-major {4,6,3,3};  A^T=[[4,6],[3,3]];  B=[16;9];  X must be [1;2].
//
// Build: icpx -fsycl -qmkl two_sycl.cpp -o two_sycl
// Run:   ./two_sycl

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int64_t n = 2, nrhs = 1, lda = 2, ldb = 2, gc = 1, gs = 1;
    auto trans = oneapi::mkl::transpose::trans;
    double A[4] = {4, 6, 3, 3};
    double B[2] = {16, 9};

    std::printf("A (col-major) = {%.4g, %.4g, %.4g, %.4g}\n", A[0], A[1], A[2], A[3]);
    std::printf("  A = [ %.4g  %.4g ]\n      [ %.4g  %.4g ]\n", A[0], A[2], A[1], A[3]);
    std::printf("B (rhs) = [ %.4g ; %.4g ]   solving A^T X = B\n", B[0], B[1]);

    double*  dA  = sycl::malloc_device<double>(4, q);
    double*  dB  = sycl::malloc_device<double>(2, q);
    int64_t* dIp = sycl::malloc_device<int64_t>(n, q);
    q.memcpy(dA, A, sizeof A);
    q.memcpy(dB, B, sizeof B);
    q.wait();

    double**  Ap  = sycl::malloc_device<double*>(1, q);
    double**  Bp  = sycl::malloc_device<double*>(1, q);
    int64_t** Ipp = sycl::malloc_device<int64_t*>(1, q);
    double* hA[1]={dA};
    double* hB[1]={dB};
    int64_t* hI[1]={dIp};
    q.memcpy(Ap, hA, sizeof hA);
    q.memcpy(Bp, hB, sizeof hB);
    q.memcpy(Ipp, hI, sizeof hI);
    q.wait();

    int64_t m_a[1]={n}, n_a[1]={n}, nr_a[1]={nrhs}, la_a[1]={lda}, lb_a[1]={ldb}, gz[1]={gs};
    oneapi::mkl::transpose tr_a[1]={trans};

    int64_t fs = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(q, m_a, n_a, la_a, gc, gz);
    double* fsc = sycl::malloc_device<double>(fs>0?fs:1, q);
    oneapi::mkl::lapack::getrf_batch(q, m_a, n_a, Ap, la_a, Ipp, gc, gz, fsc, fs, {});
    q.wait();

    // --- intermediate: LU factors and pivots after getrf, before getrs ---
    double luA[4] = {0, 0, 0, 0};
    int64_t piv[2] = {0, 0};
    q.memcpy(luA, dA, sizeof luA).wait();
    q.memcpy(piv, dIp, sizeof piv).wait();
    std::printf("after getrf: LU(A) col-major = {%.6g, %.6g, %.6g, %.6g}\n",
                luA[0], luA[1], luA[2], luA[3]);
    std::printf("after getrf: ipiv = [ %ld ; %ld ]  (1-based)\n", (long)piv[0], (long)piv[1]);

    int64_t ss = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(q, tr_a, n_a, nr_a, la_a, lb_a, gc, gz);
    double* ssc = sycl::malloc_device<double>(ss>0?ss:1, q);
    oneapi::mkl::lapack::getrs_batch(q, tr_a, n_a, nr_a,
        const_cast<const double* const*>(Ap), la_a,
        const_cast<const int64_t* const*>(Ipp), Bp, lb_a, gc, gz, ssc, ss, {});
    q.wait();

    double X[2] = {0,0};
    q.memcpy(X, dB, sizeof X).wait();
    std::printf("solution X = [ %.10g ; %.10g ]   expected [ 1 ; 2 ]\n", X[0], X[1]);
    return 0;
}
