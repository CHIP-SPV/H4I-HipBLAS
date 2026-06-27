// Minimal pure-oneMKL + native Level Zero reproducer, 2x2 transpose only.
//
// Builds the SYCL queue from a raw Level Zero command queue (mirroring how
// H4I-MKLShim Context.cpp wraps chipStar's LZ queue) instead of letting SYCL
// create its own. Solves A^T X = B for the hand-verifiable system:
//   A = [[4,3],[6,3]] (col-major {4,6,3,3}),  A^T = [[4,6],[3,3]]
//   X = [1;2]  ->  B = A^T * X = [16;9].  Must recover X = [1;2].
//
// Build:  icpx -fsycl -qmkl -lze_loader mkl_getrs_lz_2x2.cpp -o lz2x2
// Run:    ./lz2x2

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    // --- create a native Level Zero command queue ---
    zeInit(0);
    uint32_t nd = 1; ze_driver_handle_t drv;  zeDriverGet(&nd, &drv);
    uint32_t ng = 1; ze_device_handle_t dev;  zeDeviceGet(drv, &ng, &dev);
    ze_context_handle_t zctx;
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    zeContextCreate(drv, &cd, &zctx);

    // Find the compute command-queue group ordinal, exactly as chipStar does.
    uint32_t grpCount = 0;
    zeDeviceGetCommandQueueGroupProperties(dev, &grpCount, nullptr);
    std::vector<ze_command_queue_group_properties_t> grps(grpCount);
    for (auto& g : grps) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(dev, &grpCount, grps.data());
    uint32_t computeOrdinal = 0;
    for (uint32_t i = 0; i < grpCount; ++i)
        if (grps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { computeOrdinal = i; break; }

    // Command queue descriptor matching chipStar's Level0 backend:
    // compute ordinal, IN_ORDER flag, ASYNCHRONOUS mode.
    ze_command_queue_handle_t zq;
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,
                                  computeOrdinal,                  // ordinal
                                  0,                               // index
                                  ZE_COMMAND_QUEUE_FLAG_IN_ORDER,  // flags
                                  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                  ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    zeCommandQueueCreate(zctx, dev, &qd, &zq);

    // --- wrap LZ handles into a SYCL queue (out-of-order: no in_order property) ---
    auto plat = sycl::detail::make_platform((ur_native_handle_t)drv,
                                            sycl::backend::ext_oneapi_level_zero);
    sycl::device sdev;
    for (auto& d : plat.get_devices())
        if (sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d) == dev) { sdev = d; break; }
    auto sctx = sycl::detail::make_context((ur_native_handle_t)zctx, {},
                 sycl::backend::ext_oneapi_level_zero, true, {sdev});
    sycl::queue q = sycl::detail::make_queue((ur_native_handle_t)zq, false, sctx, &sdev, true,
                 {}, {}, sycl::backend::ext_oneapi_level_zero);

    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    MKLVersion v; mkl_get_version(&v);
    std::printf("oneMKL %d.%d update %d (build %s)\n",
                v.MajorVersion, v.MinorVersion, v.UpdateVersion, v.Build);

    // --- 2x2 transpose known solution ---
    const int64_t n = 2, nrhs = 1, lda = 2, ldb = 2, gc = 1, gs = 1;
    auto trans = oneapi::mkl::transpose::trans;
    double A[4] = {4, 6, 3, 3};
    double B[2] = {16, 9};

    double*  dA = sycl::malloc_device<double>(4, q);
    double*  dB = sycl::malloc_device<double>(2, q);
    int64_t* dIp = sycl::malloc_device<int64_t>(n, q);
    q.memcpy(dA, A, sizeof A); q.memcpy(dB, B, sizeof B); q.wait();

    double**  Ap  = sycl::malloc_device<double*>(1, q);
    double**  Bp  = sycl::malloc_device<double*>(1, q);
    int64_t** Ipp = sycl::malloc_device<int64_t*>(1, q);
    double* hA[1]={dA}; double* hB[1]={dB}; int64_t* hI[1]={dIp};
    q.memcpy(Ap, hA, sizeof hA); q.memcpy(Bp, hB, sizeof hB); q.memcpy(Ipp, hI, sizeof hI); q.wait();

    int64_t m_a[1]={n}, n_a[1]={n}, nr_a[1]={nrhs}, la_a[1]={lda}, lb_a[1]={ldb}, gz[1]={gs};
    oneapi::mkl::transpose tr_a[1]={trans};

    int64_t fs = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(q, m_a, n_a, la_a, gc, gz);
    double* fsc = sycl::malloc_device<double>(fs>0?fs:1, q);
    oneapi::mkl::lapack::getrf_batch(q, m_a, n_a, Ap, la_a, Ipp, gc, gz, fsc, fs, {});
    q.wait();

    int64_t ss = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(q, tr_a, n_a, nr_a, la_a, lb_a, gc, gz);
    double* ssc = sycl::malloc_device<double>(ss>0?ss:1, q);
    oneapi::mkl::lapack::getrs_batch(q, tr_a, n_a, nr_a,
        const_cast<const double* const*>(Ap), la_a,
        const_cast<const int64_t* const*>(Ipp), Bp, lb_a, gc, gz, ssc, ss, {});
    q.wait();

    double X[2] = {0,0};
    q.memcpy(X, dB, sizeof X).wait();
    std::printf("X = [%.10g ; %.10g]  expected [1 ; 2]\n", X[0], X[1]);
    double err = std::max(std::abs(X[0]-1.0), std::abs(X[1]-2.0));
    std::printf("max|X-exp| = %.3e -> %s\n", err, (err <= 1e-8) ? "PASS" : "FAIL");
    return 0;
}
