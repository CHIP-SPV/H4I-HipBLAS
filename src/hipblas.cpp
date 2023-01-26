// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklblas.h"
#include "h4i/hipblas/impl/util.h"

#define HIPBLAS_TRY \
  if (handle == nullptr) {\
    return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;\
  }\
  try {

#define HIPBLAS_CATCH(msg) \
  } catch(std::exception const& e) {\
    std::cerr <<msg<<" exception: " << e.what() << std::endl;\
    return HIPBLAS_STATUS_EXECUTION_FAILED;\
  }\
  return HIPBLAS_STATUS_SUCCESS;

// Level-1 : asum (supported datatypes : float, double, complex float, complex double)
// Generic asum which can handle batched/stride/non-batched
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
		hipError_t hip_status;
		bool is_result_dev_ptr = isDevicePointer(result);
		// 'result' can be device or host memory but oneMKL needs device memory
		float* dev_result = result;
		if (!is_result_dev_ptr) {
			hip_status = hipMalloc(&dev_result, sizeof(float));
		}

		H4I::MKLShim::sAsum(ctxt, n, x, incx, dev_result);

		if (!is_result_dev_ptr) {
			hip_status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
			hip_status = hipFree(dev_result);
		}
  HIPBLAS_CATCH("ASUM")
}

hipblasStatus_t
  hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    double* dev_result = result;
    if (!is_result_dev_ptr) {
      hip_status = hipMalloc(&dev_result, sizeof(double));
    }

    H4I::MKLShim::dAsum(ctxt, n, x, incx, dev_result);

    if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
    }
  HIPBLAS_CATCH("ASUM")
}

hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    float* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(float));
    }

    H4I::MKLShim::cAsum(ctxt, n, (const float _Complex*)x, incx, dev_result);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }
  HIPBLAS_CATCH("ASUM")
}

hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    double* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(double));
    }

    H4I::MKLShim::zAsum(ctxt, n, (const double _Complex*)x, incx, dev_result);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }
  HIPBLAS_CATCH("ASUM")
}

// asum_batched
hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDasumBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


// asum_strided_batched
hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : axpy (supported datatypes : float, double, complex float, complex double)
// Generic axpy which can handle batched/stride/non-batched
hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    bool is_result_dev_ptr = isDevicePointer(alpha);

    // 'alpha' can be device or host memory hence need to be copied before access
    float host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(float), hipMemcpyDefault);
    } else {
      host_alpha_ptr = *alpha;
    }

    H4I::MKLShim::sAxpy(ctxt, n, host_alpha_ptr, x, incx, y, incy);
  HIPBLAS_CATCH("AXPY")
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double* alpha,
                             const double* x, int incx, double* y, int incy){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    bool is_result_dev_ptr = isDevicePointer(alpha);

    // 'alpha' can be device or host memory hence need to be copied before access
    double host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(double), hipMemcpyDefault);
    } else {
      host_alpha_ptr = *alpha;
    }

    H4I::MKLShim::dAxpy(ctxt, n, host_alpha_ptr, x, incx, y, incy);
  HIPBLAS_CATCH("AXPY")
}

hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n, const hipblasComplex* alpha,
                             const hipblasComplex* x, int incx, hipblasComplex* y, int incy){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    bool is_result_dev_ptr = isDevicePointer(alpha);

    // 'alpha' can be device or host memory hence need to be copied before access
    float _Complex host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
      host_alpha_ptr = *((const float _Complex*)alpha);
    }
    H4I::MKLShim::cAxpy(ctxt, n, host_alpha_ptr, (const float _Complex*)x, incx, (float _Complex*)y, incy);
  HIPBLAS_CATCH("AXPY")
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    bool is_result_dev_ptr = isDevicePointer(alpha);

    // 'alpha' can be device or host memory hence need to be copied before access
    double _Complex host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
      host_alpha_ptr = *((const double _Complex*)alpha);
    }

    H4I::MKLShim::zAxpy(ctxt, n, host_alpha_ptr, (const double _Complex*)x, incx, (double _Complex*)y, incy);
  HIPBLAS_CATCH("AXPY")
}
// axpy_batched
hipblasStatus_t hipblasHaxpyBatched(hipblasHandle_t          handle,
                                    int                      n,
                                    const hipblasHalf*       alpha,
                                    const hipblasHalf* const x[],
                                    int                      incx,
                                    hipblasHalf* const       y[],
                                    int                      incy,
                                    int                      batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// axpy_strided_batched
hipblasStatus_t hipblasHaxpyStridedBatched(hipblasHandle_t    handle,
                                           int                n,
                                           const hipblasHalf* alpha,
                                           const hipblasHalf* x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           hipblasHalf*       y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : amax (supported datatypes : float, double, complex float, complex double)
// Generic amax which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result) {
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;
    hipError_t hip_status;
    if (!is_result_dev_ptr)
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    H4I::MKLShim::sAmax(ctxt, n, x, incx, (int64_t*)result);

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;
        hip_status = hipFree(&dev_results);
    }
  HIPBLAS_CATCH("AMAX")
}
hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    H4I::MKLShim::dAmax(ctxt, n, x, incx, dev_results);

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }
  HIPBLAS_CATCH("AMAX")
}

hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    H4I::MKLShim::cAmax(ctxt, n, (const float _Complex*)x, incx, dev_results);

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }
  HIPBLAS_CATCH("AMAX")
}

hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result){
  HIPBLAS_TRY
    auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    H4I::MKLShim::zAmax(ctxt, n, (const double _Complex*)x, incx, dev_results);

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }
  HIPBLAS_CATCH("AMAX")
}
// amax_batched
hipblasStatus_t hipblasIsamaxBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, int* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdamaxBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, int* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     int*                        result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     int*                              result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// amax_strided_batched
hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            int*                  result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            int*                        result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}