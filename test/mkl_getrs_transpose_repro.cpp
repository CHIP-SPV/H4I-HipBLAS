// Pure oneMKL reproducer for the batched transpose getrs divergence.
//
// No chipStar, no hipBLAS, no MKLShim. Just a SYCL queue + oneMKL group-API
// getrf_batch -> getrs_batch(trans=T), mirroring exactly what the shim does.
//
// Hand-verifiable known solution (matches testDgetrsKnownSolutionTranspose):
//   A = [[4,3],[6,3]] (col-major {4,6,3,3}),  A^T = [[4,6],[3,3]]
//   X = [1;2]  ->  B = A^T * X = [4*1+6*2 ; 3*1+3*2] = [16;9]
//   Solving A^T X = [16;9] must recover X = [1;2].
//
// Build (on a node, with oneAPI loaded):
//   icpx -fsycl -qmkl mkl_getrs_transpose_repro.cpp -o repro
// To pin the suspect MKL, load oneapi/2025.0.4 first; compare with 2025.3.1.
// Run:
//   CHIP_BE unused here. Just: ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./repro
//   (also try opencl:gpu to cover both backends)
//
// Prints MKL version, info, and the recovered X vs expected [1;2].

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <cstdio>
#include <vector>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- MKL version ----
    MKLVersion v;
    mkl_get_version(&v);
    std::printf("oneMKL %d.%d update %d (build %s)\n",
                v.MajorVersion, v.MinorVersion, v.UpdateVersion, v.Build);

    const int64_t n = 2, nrhs = 1, lda = 2, ldb = 2;
    const int64_t group_count = 1, group_size = 1;   // single matrix, one group
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;  // A^T X = B

    // Host data (col-major). A = [[4,3],[6,3]] -> column-major storage {4,6,3,3}.
    double A_host[4]  = {4.0, 6.0, 3.0, 3.0};
    double B_host[2]  = {16.0, 9.0};   // = A^T * [1;2]

    // ---- USM device allocations ----
    double*  dA   = sycl::malloc_device<double>(4, q);
    double*  dB   = sycl::malloc_device<double>(2, q);
    int64_t* dIpiv = sycl::malloc_device<int64_t>(n, q);
    q.memcpy(dA, A_host, 4 * sizeof(double)).wait();
    q.memcpy(dB, B_host, 2 * sizeof(double)).wait();

    // Group-API pointer arrays live in USM (MKL dereferences them on device),
    // mirroring how the shim builds ipiv_ptrs_device etc.
    double**  A_ptrs    = sycl::malloc_shared<double*>(group_size, q);
    double**  B_ptrs    = sycl::malloc_shared<double*>(group_size, q);
    int64_t** ipiv_ptrs = sycl::malloc_shared<int64_t*>(group_size, q);
    A_ptrs[0] = dA; B_ptrs[0] = dB; ipiv_ptrs[0] = dIpiv;

    // Group arrays (host arrays, read by MKL). Keep alive until waits below.
    int64_t m_arr[1]    = {n};
    int64_t n_arr[1]    = {n};
    int64_t nrhs_arr[1] = {nrhs};
    int64_t lda_arr[1]  = {lda};
    int64_t ldb_arr[1]  = {ldb};
    int64_t gsz[1]      = {group_size};
    oneapi::mkl::transpose trans_arr[1] = {trans};

    try {
        // ---- getrf_batch (group) ----
        int64_t f_scr = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(
            q, m_arr, n_arr, lda_arr, group_count, gsz);
        double* f_scratch = sycl::malloc_device<double>(f_scr > 0 ? f_scr : 1, q);

        auto ef = oneapi::mkl::lapack::getrf_batch(
            q, m_arr, n_arr, A_ptrs, lda_arr, ipiv_ptrs,
            group_count, gsz, f_scratch, f_scr, {});
        q.wait();

        // ---- getrs_batch (group, trans=T) ----
        int64_t s_scr = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(
            q, trans_arr, n_arr, nrhs_arr, lda_arr, ldb_arr, group_count, gsz);
        double* s_scratch = sycl::malloc_device<double>(s_scr > 0 ? s_scr : 1, q);

        auto es = oneapi::mkl::lapack::getrs_batch(
            q, trans_arr, n_arr, nrhs_arr,
            const_cast<const double* const*>(A_ptrs), lda_arr,
            const_cast<const int64_t* const*>(ipiv_ptrs),
            B_ptrs, ldb_arr, group_count, gsz, s_scratch, s_scr, {});
        q.wait();

        sycl::free(f_scratch, q);
        sycl::free(s_scratch, q);
    } catch (oneapi::mkl::lapack::exception const& e) {
        std::printf("MKL LAPACK exception: info=%ld detail=%ld what=%s\n",
                    (long)e.info(), (long)e.detail(), e.what());
    } catch (sycl::exception const& e) {
        std::printf("SYCL exception: %s\n", e.what());
    }

    // ---- read back X ----
    double X[2] = {0, 0};
    q.memcpy(X, dB, 2 * sizeof(double)).wait();

    std::printf("\nResult: X = [%.10g ; %.10g]   expected [1 ; 2]\n", X[0], X[1]);
    double err = std::max(std::abs(X[0] - 1.0), std::abs(X[1] - 2.0));
    std::printf("max|X-expected| = %.3e  -> %s\n", err, (err < 1e-8 ? "PASS" : "FAIL"));

    // Reference: also show what the trans=N answer WOULD be, to detect a flag-ignored bug.
    // Solve A X = [16;9]: X = A^{-1} b. A=[[4,3],[6,3]], det = 4*3-3*6 = -6.
    // A^{-1} = (1/det)[[3,-3],[-6,4]] = [[-0.5,0.5],[1,-0.6667]].
    // X_N = A^{-1}[16;9] = [-0.5*16+0.5*9 ; 1*16-0.6667*9] = [-3.5 ; 10].
    std::printf("(for reference, trans=N solve A X=[16;9] would give X=[-3.5 ; 10])\n");

    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dIpiv, q);
    sycl::free(A_ptrs, q); sycl::free(B_ptrs, q); sycl::free(ipiv_ptrs, q);
    return 0;
}
