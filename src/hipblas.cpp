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

// TODO: These are utility functions which can be moved to other files
H4I::MKLShim::onemklTranspose convert(hipblasOperation_t val) {
    switch(val) {
        case HIPBLAS_OP_T:
            return H4I::MKLShim::ONEMKL_TRANSPOSE_TRANS;
        case HIPBLAS_OP_C:
            return H4I::MKLShim::ONEMLK_TRANSPOSE_CONJTRANS;
        case HIPBLAS_OP_N:
        default:
            return H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS;
    }
}

H4I::MKLShim::onemklUplo convert(hipblasFillMode_t val) {
    switch(val) {
        case HIPBLAS_FILL_MODE_UPPER:
            return H4I::MKLShim::ONEMKL_UPLO_UPPER;
        case HIPBLAS_FILL_MODE_LOWER:
            return H4I::MKLShim::ONEMKL_UPLO_LOWER;
    }
}

H4I::MKLShim::onemklDiag convert(hipblasDiagType_t val) {
    switch(val) {
        case HIPBLAS_DIAG_NON_UNIT:
            return H4I::MKLShim::ONEMKL_DIAG_NONUNIT;
        case HIPBLAS_DIAG_UNIT:
            return H4I::MKLShim::ONEMKL_DIAG_UNIT;
    }
}

H4I::MKLShim::onemklSideMode convert(hipblasSideMode_t val) {
    switch(val) {
        case HIPBLAS_SIDE_LEFT:
            return H4I::MKLShim::ONEMKL_SIDE_LEFT;
        case HIPBLAS_SIDE_RIGHT:
            return H4I::MKLShim::ONEMKL_SIDE_RIGHT;
    }
}

// Level-1 : asum (supported datatypes : float, double, complex float, complex double)
// Generic asum which can handle batched/stride/non-batched
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result){
  HIPBLAS_TRY
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  bool is_result_dev_ptr = isDevicePointer(result);
  // special cases
  if (incx <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  } else if (n <= 0) {
    if (is_result_dev_ptr) {
      hip_status = hipMemset(result, 0, sizeof(result));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }
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
  // special cases
  if (incx <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  } else if (n <= 0) {
    if (is_result_dev_ptr) {
      hip_status = hipMemset(result, 0, sizeof(result));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }
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
  // special cases
  if (incx <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  } else if (n <= 0) {
    if (is_result_dev_ptr) {
      hip_status = hipMemset(result, 0, sizeof(result));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }
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
  // special cases
  if (incx <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  } else if (n <= 0) {
    if (is_result_dev_ptr) {
      hip_status = hipMemset(result, 0, sizeof(result));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }
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
hipblasStatus_t hipblasHaxpy(hipblasHandle_t handle, int n, const hipblasHalf* alpha,
                             const hipblasHalf* x,int incx, hipblasHalf* y, int incy) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

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
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::sAmax(ctxt, n, x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMAX")
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::dAmax(ctxt, n, x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMAX")
}

hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::cAmax(ctxt, n, (const float _Complex*)x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMAX")
}

hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::zAmax(ctxt, n, (const double _Complex*)x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
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

// Level-1 : amin (supported datatypes : float, double, complex float, complex double)
// Generic amin which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::sAmin(ctxt, n, x, incx, dev_results);
  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMIN")
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
	  return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::dAmin(ctxt, n, x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMIN")
}

hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::cAmin(ctxt, n, (const float _Complex*)x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMIN")
}

hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result){
  HIPBLAS_TRY
  hipError_t hip_status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (x == nullptr) {
    if (is_result_dev_ptr){
      hipMemset(result, 0, sizeof(int));
    } else {
      *result = 0;
    }
    return HIPBLAS_STATUS_SUCCESS;
  }

  // error checks
  if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int64_t *dev_results = nullptr;
  hip_status = hipMalloc(&dev_results, sizeof(int64_t));

  H4I::MKLShim::zAmin(ctxt, n, (const double _Complex*)x, incx, dev_results);

  // Workaround:
  // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
  // handling it below by incrementing it by one
  // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
  int64_t results_host_memory = 0;
  hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
  if (!H4I::MKLShim::is_mkl_eq_higher_2023_0_2()) {
    results_host_memory += 1;
  }

  int return_val = (int)results_host_memory;
  if (is_result_dev_ptr) {
    hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
  } else {
    *result = return_val;
  }
  hip_status = hipFree(&dev_results);
  HIPBLAS_CATCH("AMIN")
}

// amin_batched
hipblasStatus_t hipblasIsaminBatched(
  hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, int* result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdaminBatched(
  hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, int* result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     int*                        result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     int*                              result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// amin_strided_batched
hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            int*                  result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            int*                        result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : copy (supported datatypes : float, double, complex float, complex double)
// Generic copy which can handle batched/stride/non-batched
hipblasStatus_t
  hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
    incx <= 0 || incy <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sCopy(ctxt, n, x, incx, y, incy);
  HIPBLAS_CATCH("COPY")
}

hipblasStatus_t
  hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
    incx <= 0 || incy <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
	H4I::MKLShim::dCopy(ctxt, n, x, incx, y, incy);
  HIPBLAS_CATCH("COPY")
}

hipblasStatus_t
  hipblasCcopy(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
    incx <= 0 || incy <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cCopy(ctxt, n, (const float _Complex*)x, incx, (float _Complex*)y, incy);
  HIPBLAS_CATCH("COPY")
}

hipblasStatus_t
  hipblasZcopy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
    incx <= 0 || incy <= 0 || n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zCopy(ctxt, n, (const double _Complex*)x, incx, (double _Complex*)y, incy);
  HIPBLAS_CATCH("COPY")
}

// copy_batched
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// copy_strided_batched
hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
// Level-1 : dot (supported datatypes : float, double, complex float, complex double)
// Generic dot which can handle batched/stride/non-batched
hipblasStatus_t hipblasHdot(hipblasHandle_t handle, int n, const hipblasHalf* x, int incx,
                            const hipblasHalf* y, int incy, hipblasHalf* result) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasBfdot(hipblasHandle_t handle, int n, const hipblasBfloat16* x,
                             int incx, const hipblasBfloat16* y, int incy, hipblasBfloat16* result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }    
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float* dev_result = result;
  if (!is_result_dev_ptr) {
    hip_status = hipMalloc(&dev_result, sizeof(float));
  }
  H4I::MKLShim::sDot(ctxt, n, x, incx, y, incy, dev_result);

  if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("DOT")
}

hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }    
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double* dev_result = result;
  if (!is_result_dev_ptr) {
      hip_status = hipMalloc(&dev_result, sizeof(double));
  }
  H4I::MKLShim::dDot(ctxt, n, x, incx, y, incy, dev_result);

  if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("DOT")
}

hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n, const hipblasComplex* x,
                             int incx, const hipblasComplex* y, int incy, hipblasComplex* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }    
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex* dev_result = (float _Complex*)result;
  if (!is_result_dev_ptr) {
      hip_status = hipMalloc(&dev_result, sizeof(float _Complex));
  }
  H4I::MKLShim::cDotc(ctxt, n, (const float _Complex*)x, incx, (const float _Complex*)y, incy, dev_result);

  if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(float _Complex), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("DOT")
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n, const hipblasComplex* x,
                             int incx, const hipblasComplex* y, int incy, hipblasComplex* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }    
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex* dev_result = (float _Complex*)result;
  if (!is_result_dev_ptr) {
      hip_status = hipMalloc(&dev_result, sizeof(float _Complex));
  }
  H4I::MKLShim::cDotu(ctxt, n, (const float _Complex*)x, incx, (const float _Complex*)y, incy, dev_result);

  if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(float _Complex), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("DOT")
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x,
                             int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }    
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex* dev_result = (double _Complex*)result;
  if (!is_result_dev_ptr) {
      hip_status = hipMalloc(&dev_result, sizeof(double _Complex));
  }
  H4I::MKLShim::zDotc(ctxt, n, (const double _Complex*)x, incx, (const double _Complex*)y, incy, dev_result);

  if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(double _Complex), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("DOT")
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x,
                             int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }    
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex* dev_result = (double _Complex*)result;
  if (!is_result_dev_ptr) {
      hip_status = hipMalloc(&dev_result, sizeof(double _Complex));
  }
  H4I::MKLShim::zDotu(ctxt, n, (const double _Complex*)x, incx, (const double _Complex*)y, incy, dev_result);

  if (!is_result_dev_ptr) {
      hip_status = hipMemcpy(result, dev_result, sizeof(double _Complex), hipMemcpyDefault);
      hip_status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("DOT")
}
// dot_batched
hipblasStatus_t hipblasHdotBatched(hipblasHandle_t          handle,
                                   int                      n,
                                   const hipblasHalf* const x[],
                                   int                      incx,
                                   const hipblasHalf* const y[],
                                   int                      incy,
                                   int                      batchCount,
                                   hipblasHalf*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasBfdotBatched(hipblasHandle_t              handle,
                                    int                          n,
                                    const hipblasBfloat16* const x[],
                                    int                          incx,
                                    const hipblasBfloat16* const y[],
                                    int                          incy,
                                    int                          batchCount,
                                    hipblasBfloat16*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                   int                n,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   int                batchCount,
                                   float*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdotBatched(hipblasHandle_t     handle,
                                   int                 n,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   int                 batchCount,
                                   double*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batchCount,
                                    hipblasComplex*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batchCount,
                                    hipblasComplex*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batchCount,
                                    hipblasDoubleComplex*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batchCount,
                                    hipblasDoubleComplex*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}


// dot_strided_batched
hipblasStatus_t hipblasHdotStridedBatched(hipblasHandle_t    handle,
                                          int                n,
                                          const hipblasHalf* x,
                                          int                incx,
                                          hipblasStride      stridex,
                                          const hipblasHalf* y,
                                          int                incy,
                                          hipblasStride      stridey,
                                          int                batchCount,
                                          hipblasHalf*       result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                           int                    n,
                                           const hipblasBfloat16* x,
                                           int                    incx,
                                           hipblasStride          stridex,
                                           const hipblasBfloat16* y,
                                           int                    incy,
                                           hipblasStride          stridey,
                                           int                    batchCount,
                                           hipblasBfloat16*       result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const float*    x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batchCount,
                                          float*          result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const double*   x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const double*   y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batchCount,
                                          double*         result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount,
                                           hipblasComplex*       result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount,
                                           hipblasComplex*       result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount,
                                           hipblasDoubleComplex*       result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount,
                                           hipblasDoubleComplex*       result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : nrm2 (supported datatypes : float, double, complex float, complex double)
// Generic nrm2 which can handle batched/stride/non-batched
hipblasStatus_t
  hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || result == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float* dev_result = result;
  if (!is_result_dev_ptr) {
      status = hipMalloc(&dev_result, sizeof(float));
  }
  H4I::MKLShim::sNrm2(ctxt, n, x, incx, dev_result);
  if (!is_result_dev_ptr) {
      status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
      status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("NRM2")
}

hipblasStatus_t
  hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || result == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double* dev_result = result;
  if (!is_result_dev_ptr) {
      status = hipMalloc(&dev_result, sizeof(double));
  }
  H4I::MKLShim::dNrm2(ctxt, n, x, incx, dev_result);
  if (!is_result_dev_ptr) {
      status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
      status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("NRM2")
}

hipblasStatus_t
  hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || result == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float* dev_result = result;
  if (!is_result_dev_ptr) {
      status = hipMalloc(&dev_result, sizeof(float));
  }
  H4I::MKLShim::cNrm2(ctxt, n, (const float _Complex*)x, incx, dev_result);
  if (!is_result_dev_ptr) {
      status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
      status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("NRM2")
}

hipblasStatus_t
  hipblasDznrm2(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || result == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_result_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double* dev_result = result;
  if (!is_result_dev_ptr) {
      status = hipMalloc(&dev_result, sizeof(double));
  }
  H4I::MKLShim::zNrm2(ctxt, n, (const double _Complex*)x, incx, dev_result);

  if (!is_result_dev_ptr) {
      status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
      status = hipFree(dev_result);
  }
  HIPBLAS_CATCH("NRM2")
}
// nrm2_batched
hipblasStatus_t hipblasSnrm2Batched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// nrm2_strided_batched
hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : rot (supported datatypes : float, double, complex float, complex double)
// Generic rot which can handle batched/stride/non-batched
hipblasStatus_t hipblasSrot(hipblasHandle_t handle,int n, float* x,int incx,
                                           float* y, int incy,const float* c, const float* s){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  float h_c, h_s;
  if (is_ptr_mode_device) {
    hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
    hip_status = hipMemcpy(&h_s, s, sizeof(float), hipMemcpyDefault);
  } else {
    h_c = *c;
    h_s = *s;
  }

  H4I::MKLShim::sRot(ctxt, n, x, incx, y, incy, h_c, h_s);
  HIPBLAS_CATCH("ROT")
}

hipblasStatus_t hipblasDrot(hipblasHandle_t handle,int n, double* x,int incx,
                                           double* y, int incy,const double* c, const double* s){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  double h_c, h_s;
  if (is_ptr_mode_device) {
    hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
    hip_status = hipMemcpy(&h_s, s, sizeof(double), hipMemcpyDefault);
  } else {
    h_c = *c;
    h_s = *s;
  }

  H4I::MKLShim::dRot(ctxt, n, x, incx, y, incy, h_c, h_s);
  HIPBLAS_CATCH("ROT")
}

hipblasStatus_t hipblasCrot(hipblasHandle_t handle,int n, hipblasComplex* x,int incx,
                                           hipblasComplex* y, int incy,const float* c,
                                           const hipblasComplex* s){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  float h_c;
  float _Complex h_s;
  if (is_ptr_mode_device) {
    hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
    hip_status = hipMemcpy(&h_s, s, sizeof(float _Complex), hipMemcpyDefault);
  } else {
    h_c = *c;
    h_s = *((float _Complex*)s);
  }

  H4I::MKLShim::cRot(ctxt, n, (float _Complex*)x, incx, (float _Complex*)y, incy, h_c, h_s);
  HIPBLAS_CATCH("ROT")
}

hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,int n, hipblasComplex* x,int incx,
                                           hipblasComplex* y, int incy,const float* c,
                                           const float* s){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  float h_c, h_s;
  if (is_ptr_mode_device) {
      hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
      hip_status = hipMemcpy(&h_s, s, sizeof(float), hipMemcpyDefault);
  } else {
      h_c = *c;
      h_s = *s;
  }

  H4I::MKLShim::csRot(ctxt, n, (float _Complex*)x, incx, (float _Complex*)y, incy, h_c, h_s);
  HIPBLAS_CATCH("ROT")
}

hipblasStatus_t hipblasZrot(hipblasHandle_t handle,int n, hipblasDoubleComplex* x,int incx,
                                           hipblasDoubleComplex* y, int incy,const double* c,
                                           const hipblasDoubleComplex* s){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  double h_c;
  double _Complex h_s;
  if (is_ptr_mode_device) {
      hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
      hip_status = hipMemcpy(&h_s, s, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_c = *c;
      h_s = *((double _Complex*)s);
  }

  H4I::MKLShim::zRot(ctxt, n, (double _Complex*)x, incx, (double _Complex*)y, incy, h_c, h_s);
  HIPBLAS_CATCH("ROT")
}

hipblasStatus_t hipblasZdrot(hipblasHandle_t handle,int n, hipblasDoubleComplex* x,int incx,
                                           hipblasDoubleComplex* y, int incy, const double* c,
                                           const double* s){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  hipError_t hip_status;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  double h_c, h_s;
  if (is_ptr_mode_device) {
      hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
      hip_status = hipMemcpy(&h_s, s, sizeof(double), hipMemcpyDefault);
  } else {
      h_c = *c;
      h_s = *s;
  }

  H4I::MKLShim::zdRot(ctxt, n, (double _Complex*)x, incx, (double _Complex*)y, incy, h_c, h_s);
  HIPBLAS_CATCH("ROT")
}
// rot_batched
hipblasStatus_t hipblasSrotBatched(hipblasHandle_t handle,
                                   int             n,
                                   float* const    x[],
                                   int             incx,
                                   float* const    y[],
                                   int             incy,
                                   const float*    c,
                                   const float*    s,
                                   int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotBatched(hipblasHandle_t handle,
                                   int             n,
                                   double* const   x[],
                                   int             incx,
                                   double* const   y[],
                                   int             incy,
                                   const double*   c,
                                   const double*   s,
                                   int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotBatched(hipblasHandle_t       handle,
                                   int                   n,
                                   hipblasComplex* const x[],
                                   int                   incx,
                                   hipblasComplex* const y[],
                                   int                   incy,
                                   const float*          c,
                                   const hipblasComplex* s,
                                   int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsrotBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    hipblasComplex* const y[],
                                    int                   incy,
                                    const float*          c,
                                    const float*          s,
                                    int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotBatched(hipblasHandle_t             handle,
                                   int                         n,
                                   hipblasDoubleComplex* const x[],
                                   int                         incx,
                                   hipblasDoubleComplex* const y[],
                                   int                         incy,
                                   const double*               c,
                                   const hipblasDoubleComplex* s,
                                   int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdrotBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    hipblasDoubleComplex* const y[],
                                    int                         incy,
                                    const double*               c,
                                    const double*               s,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rot_strided_batched
hipblasStatus_t hipblasSrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          float*          x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          float*          y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          const float*    c,
                                          const float*    s,
                                          int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          double*         x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          double*         y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          const double*   c,
                                          const double*   s,
                                          int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t       handle,
                                          int                   n,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       y,
                                          int                   incy,
                                          hipblasStride         stridey,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const float*    c,
                                           const float*    s,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t             handle,
                                          int                         n,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       y,
                                          int                         incy,
                                          hipblasStride               stridey,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           const double*         c,
                                           const double*         s,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : rotg (supported datatypes : float, double, complex float, complex double)
// Generic rotg which can handle batched/stride/non-batched
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s){
  HIPBLAS_TRY
  // error check
  if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  // FixMe: oneAPI supports only device pointers
  if (!is_ptr_mode_device) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  H4I::MKLShim::sRotg(ctxt, a, b, c, s);
  HIPBLAS_CATCH("ROTG")
}

hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s){
  HIPBLAS_TRY
  // error check
  if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  // FixMe: oneAPI supports only device pointers
  if (!is_ptr_mode_device) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  H4I::MKLShim::dRotg(ctxt, a, b, c, s);
  HIPBLAS_CATCH("ROTG")
}

hipblasStatus_t hipblasCrotg(hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s){
  HIPBLAS_TRY
  // error check
  if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  // FixMe: oneAPI supports only device pointers
  if (!is_ptr_mode_device) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  H4I::MKLShim::cRotg(ctxt, (float _Complex*)a, (float _Complex*)b, c, (float _Complex*)s);
  HIPBLAS_CATCH("ROTG")
}

hipblasStatus_t hipblasZrotg(hipblasHandle_t handle, hipblasDoubleComplex* a,
                             hipblasDoubleComplex* b, double* c, hipblasDoubleComplex* s){
  HIPBLAS_TRY
  // error check
  if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  // FixMe: oneAPI supports only device pointers
  if (!is_ptr_mode_device) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  H4I::MKLShim::zRotg(ctxt, (double _Complex*)a, (double _Complex*)b, c, (double _Complex*)s);
  HIPBLAS_CATCH("ROTG")
}
// rotg_batched
hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle,
                                    float* const    a[],
                                    float* const    b[],
                                    float* const    c[],
                                    float* const    s[],
                                    int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle,
                                    double* const   a[],
                                    double* const   b[],
                                    double* const   c[],
                                    double* const   s[],
                                    int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t       handle,
                                    hipblasComplex* const a[],
                                    hipblasComplex* const b[],
                                    float* const          c[],
                                    hipblasComplex* const s[],
                                    int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t             handle,
                                    hipblasDoubleComplex* const a[],
                                    hipblasDoubleComplex* const b[],
                                    double* const               c[],
                                    hipblasDoubleComplex* const s[],
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotg_strided_batched
hipblasStatus_t hipblasSrotgStridedBatched(hipblasHandle_t handle,
                                           float*          a,
                                           hipblasStride   stride_a,
                                           float*          b,
                                           hipblasStride   stride_b,
                                           float*          c,
                                           hipblasStride   stride_c,
                                           float*          s,
                                           hipblasStride   stride_s,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle,
                                           double*         a,
                                           hipblasStride   stride_a,
                                           double*         b,
                                           hipblasStride   stride_b,
                                           double*         c,
                                           hipblasStride   stride_c,
                                           double*         s,
                                           hipblasStride   stride_s,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle,
                                           hipblasComplex* a,
                                           hipblasStride   stride_a,
                                           hipblasComplex* b,
                                           hipblasStride   stride_b,
                                           float*          c,
                                           hipblasStride   stride_c,
                                           hipblasComplex* s,
                                           hipblasStride   stride_s,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t       handle,
                                           hipblasDoubleComplex* a,
                                           hipblasStride         stride_a,
                                           hipblasDoubleComplex* b,
                                           hipblasStride         stride_b,
                                           double*               c,
                                           hipblasStride         stride_c,
                                           hipblasDoubleComplex* s,
                                           hipblasStride         stride_s,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : rotm (supported datatypes : float, double)
// Generic rotm which can handle batched/stride/non-batched
hipblasStatus_t hipblasSrotm(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0 ||x == nullptr || y == nullptr ||param == nullptr) {
      return HIPBLAS_STATUS_SUCCESS;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hipStatus;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  float* dev_param = (float*) param;
  if (!is_ptr_mode_device) {
      hipStatus = hipMalloc(&dev_param, sizeof(float)*5);
      hipStatus = hipMemcpy(dev_param, param, sizeof(float)*5, hipMemcpyHostToDevice);
  }
  H4I::MKLShim::sRotm(ctxt, n, x, incx, y, incy, dev_param);
  if (!is_ptr_mode_device) {
      hipStatus = hipFree(dev_param);
  }
  HIPBLAS_CATCH("ROTM")
}

hipblasStatus_t hipblasDrotm(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0 || x == nullptr || y == nullptr ||param == nullptr) {
      return HIPBLAS_STATUS_SUCCESS;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipError_t hipStatus;
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  double* dev_param = (double*)param;
  if (!is_ptr_mode_device) {
      hipStatus = hipMalloc(&dev_param, sizeof(double)*5);
      hipStatus = hipMemcpy(dev_param, param, sizeof(double)*5, hipMemcpyHostToDevice);
  }
  H4I::MKLShim::dRotm(ctxt, n, x, incx, y, incy, dev_param);
  if (!is_ptr_mode_device) {
      hipStatus = hipFree(dev_param);
  }
  HIPBLAS_CATCH("ROTM")
}
// rotm_batched
hipblasStatus_t hipblasSrotmBatched(hipblasHandle_t    handle,
                                    int                n,
                                    float* const       x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    const float* const param[],
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    double* const       x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    const double* const param[],
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotm_strided_batched
hipblasStatus_t hipblasSrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const float*    param,
                                           hipblasStride   strideParam,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const double*   param,
                                           hipblasStride   strideParam,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : rotmg(supported datatypes : float and double )
hipblasStatus_t hipblasSrotmg(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param) {
  HIPBLAS_TRY
  if (handle == nullptr || d1 == nullptr || d2 == nullptr || x1 == nullptr || y1 == nullptr || param == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (!is_ptr_mode_device) {
    // MKL does not support host pointers
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  // MKL takes y1 as scalar not a pointer hence need to have a host copy to access it incase of device pointer
  float host_y1 = 1.0;
  if (is_ptr_mode_device) {
      auto hipStatus = hipMemcpy(&host_y1, y1, sizeof(float), hipMemcpyDefault);
  }
  H4I::MKLShim::sRotmg(ctxt, d1, d2, x1, host_y1, param);
  HIPBLAS_CATCH("ROTMG")
}

hipblasStatus_t hipblasDrotmg(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param) {
  HIPBLAS_TRY
  if (handle == nullptr || d1 == nullptr || d2 == nullptr || x1 == nullptr || y1 == nullptr || param == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_ptr_mode_device = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  if (!is_ptr_mode_device) {
    // MKL does not support host pointers
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  // MKL takes y1 as scalar not a pointer hence need to have a host copy to access it incase of device pointer
  double host_y1 = 1.0;
  if (is_ptr_mode_device) {
      auto hipStatus = hipMemcpy(&host_y1, y1, sizeof(double), hipMemcpyDefault);
  }
  H4I::MKLShim::dRotmg(ctxt, d1, d2, x1, host_y1, param);
  HIPBLAS_CATCH("ROTMG")
}

// rotmg_batched
hipblasStatus_t hipblasSrotmgBatched(hipblasHandle_t    handle,
                                     float* const       d1[],
                                     float* const       d2[],
                                     float* const       x1[],
                                     const float* const y1[],
                                     float* const       param[],
                                     int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t     handle,
                                     double* const       d1[],
                                     double* const       d2[],
                                     double* const       x1[],
                                     const double* const y1[],
                                     double* const       param[],
                                     int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotmg_strided_batched
hipblasStatus_t hipblasSrotmgStridedBatched(hipblasHandle_t handle,
                                            float*          d1,
                                            hipblasStride   stride_d1,
                                            float*          d2,
                                            hipblasStride   stride_d2,
                                            float*          x1,
                                            hipblasStride   stride_x1,
                                            const float*    y1,
                                            hipblasStride   stride_y1,
                                            float*          param,
                                            hipblasStride   strideParam,
                                            int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle,
                                            double*         d1,
                                            hipblasStride   stride_d1,
                                            double*         d2,
                                            hipblasStride   stride_d2,
                                            double*         x1,
                                            hipblasStride   stride_x1,
                                            const double*   y1,
                                            hipblasStride   stride_y1,
                                            double*         param,
                                            hipblasStride   strideParam,
                                            int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : scal (supported datatypes : float, double, complex float, complex double)
// Generic scal which can handle batched/stride/non-batched
hipblasStatus_t
    hipblasSscal(hipblasHandle_t handle, int n, const float *alpha, float *x, int incx){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || alpha == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  float host_alpha = 1.0;
  if (is_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      host_alpha = *alpha;
  }

  H4I::MKLShim::sScal(ctxt, n, host_alpha, x, incx);
  HIPBLAS_CATCH("SCAL")
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double *alpha, double *x, int incx){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || alpha == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  double host_alpha = 1.0;
  if (is_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      host_alpha = *alpha;
  }

  H4I::MKLShim::dScal(ctxt, n, host_alpha, x, incx);
  HIPBLAS_CATCH("SCAL")
}

hipblasStatus_t
    hipblasCscal(hipblasHandle_t handle, int n, const hipblasComplex *alpha, hipblasComplex *x, int incx){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || alpha == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  float _Complex host_alpha = 1.0;
  if (is_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      host_alpha = *((float _Complex*)alpha);
  }

  H4I::MKLShim::cScal(ctxt, n, host_alpha, (float _Complex*)x, incx);
  HIPBLAS_CATCH("SCAL")
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float *alpha, hipblasComplex *x, int incx){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || alpha == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  float host_alpha = 1.0;
  if (is_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(float ), hipMemcpyDefault);
  } else {
      host_alpha = *alpha;
  }
  H4I::MKLShim::csScal(ctxt, n, host_alpha, (float _Complex*)x, incx);
  HIPBLAS_CATCH("SCAL")
}

hipblasStatus_t
    hipblasZscal(hipblasHandle_t handle, int n, const hipblasDoubleComplex *alpha, hipblasDoubleComplex *x, int incx){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || alpha == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  double _Complex host_alpha = 1.0;
  if (is_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      host_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zsScal(ctxt, n, host_alpha, (double _Complex*)x, incx);
  HIPBLAS_CATCH("SCAL")
}

hipblasStatus_t
    hipblasZdscal(hipblasHandle_t handle, int n, const double *alpha, hipblasDoubleComplex *x, int incx){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || alpha == nullptr ||
      incx <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  double host_alpha = 1.0;
  if (is_dev_ptr) {
      auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      host_alpha = *alpha;
  }
  H4I::MKLShim::zdScal(ctxt, n, host_alpha, (double _Complex*)x, incx);
  HIPBLAS_CATCH("SCAL")
}
// scal_batched
hipblasStatus_t hipblasSscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, double* const x[], int incx, int batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t       handle,
                                     int                   n,
                                     const float*          alpha,
                                     hipblasComplex* const x[],
                                     int                   incx,
                                     int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const double*               alpha,
                                     hipblasDoubleComplex* const x[],
                                     int                         incx,
                                     int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// scal_strided_batched
hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    alpha,
                                            hipblasComplex* x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const double*         alpha,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-1 : swap (supported datatypes : float, double, complex float, complex double)
// Generic swap which can handle batched/stride/non-batched
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sSwap(ctxt, n, x, incx, y, incy);
  HIPBLAS_CATCH("SWAP")
}

hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dSwap(ctxt, n, x, incx, y, incy);
  HIPBLAS_CATCH("SWAP")
}

hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n, hipblasComplex* x, int incx,
                             hipblasComplex* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cSwap(ctxt, n, (float _Complex*)x, incx, (float _Complex*)y, incy);
  HIPBLAS_CATCH("SWAP")
}

hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx,
                             hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  // error checks
  if (handle == nullptr || x == nullptr || y == nullptr ||
      incx <= 0 || incy <= 0 || n <= 0) {
      return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zSwap(ctxt, n, (double _Complex*)x, incx, (double _Complex*)y, incy);
  HIPBLAS_CATCH("SWAP")
}
// swap_batched
hipblasStatus_t hipblasSswapBatched(
    hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDswapBatched(
    hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle,
                                    int             n,
                                    hipblasComplex* x[],
                                    int             incx,
                                    hipblasComplex* y[],
                                    int             incy,
                                    int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasDoubleComplex* x[],
                                    int                   incx,
                                    hipblasDoubleComplex* y[],
                                    int                   incy,
                                    int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// swap_strided_batched
hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// Level-2 : gbmv(supported datatypes : float, double, float complex and doule complex)
hipblasStatus_t hipblasSgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const float* alpha,
                              const float* AP, int lda, const float* x, int incx,
                              const float* beta, float* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  // Need to check as alpha and beta can be host/device pointer
  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sGbmv(ctxt, convert(trans), m, n, kl, ku, h_alpha, AP, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("GBMV")
}

hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const double* alpha,
                              const double* AP, int lda, const double* x, int incx,
                              const double* beta, double* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dGbmv(ctxt, convert(trans), m, n, kl, ku, h_alpha, AP, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("GBMV")
}

hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const hipblasComplex* alpha,
                              const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx,
                              const hipblasComplex* beta, hipblasComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cGbmv(ctxt, convert(trans), m, n, kl, ku, h_alpha,
              (const float _Complex *)AP, lda, (const float _Complex *)x, incx,
                h_beta, (float _Complex *)y, incy);
  HIPBLAS_CATCH("GBMV")
}

hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx,
                              const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zGbmv(ctxt, convert(trans), m, n, kl, ku, h_alpha,
              (const double _Complex *)AP, lda, (const double _Complex *)x, incx,
                h_beta, (double _Complex *)y, incy);
  HIPBLAS_CATCH("GBMV")
}

// gbmv_batched
hipblasStatus_t hipblasSgbmvBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    int                m,
                                    int                n,
                                    int                kl,
                                    int                ku,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float* const       y[],
                                    int                incy,
                                    int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgbmvBatched(hipblasHandle_t     handle,
                                    hipblasOperation_t  trans,
                                    int                 m,
                                    int                 n,
                                    int                 kl,
                                    int                 ku,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgbmvBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          trans,
                                    int                         m,
                                    int                         n,
                                    int                         kl,
                                    int                         ku,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgbmvBatched(hipblasHandle_t                   handle,
                                    hipblasOperation_t                trans,
                                    int                               m,
                                    int                               n,
                                    int                               kl,
                                    int                               ku,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gbmv_strided_batched
hipblasStatus_t hipblasSgbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           int                kl,
                                           int                ku,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           const float*       x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           const float*       beta,
                                           float*             y,
                                           int                incy,
                                           hipblasStride      stride_y,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           int                kl,
                                           int                ku,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           const double*      x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           const double*      beta,
                                           double*            y,
                                           int                incy,
                                           hipblasStride      stride_y,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           int                   m,
                                           int                   n,
                                           int                   kl,
                                           int                   ku,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stride_y,
                                           int                   batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          trans,
                                           int                         m,
                                           int                         n,
                                           int                         kl,
                                           int                         ku,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stride_y,
                                           int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : gemv(supported datatypes : float, double, float complex and doule complex)
hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                             const float* alpha, const float* AP, int lda, const float* x, int incx,
                             const float* beta, float* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sGemv(ctxt, convert(trans), m, n, h_alpha, AP, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("GEMV")
}

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                             const double* alpha, const double* AP, int lda, const double* x, int incx,
                             const double* beta, double* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dGemv(ctxt, convert(trans), m, n, h_alpha, AP, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("GEMV")
}

hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, const hipblasComplex* alpha,
                              const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx,
                              const hipblasComplex* beta, hipblasComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cGemv(ctxt, convert(trans), m, n, h_alpha,
              (const float _Complex *)AP, lda, (const float _Complex *)x, incx,
                h_beta, (float _Complex *)y, incy);
  HIPBLAS_CATCH("GEMV")
}

hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx,
                              const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zGemv(ctxt, convert(trans), m, n, h_alpha,
              (const double _Complex *)AP, lda, (const double _Complex *)x, incx,
                h_beta, (double _Complex *)y, incy);
  HIPBLAS_CATCH("GEMV")
}

// gemv_batched
hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgemvBatched(hipblasHandle_t     handle,
                                    hipblasOperation_t  trans,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgemvBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          trans,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvBatched(hipblasHandle_t                   handle,
                                    hipblasOperation_t                trans,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gemv_strided_batched
hipblasStatus_t hipblasSgemvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const float*       x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           const float*       beta,
                                           float*             y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const double*      x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           const double*      beta,
                                           double*            y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          trans,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : ger(supported datatypes : float, double, float complex and doule complex)
hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n, const float* alpha,
                            const float* x, int incx, const float* y, int incy,
                            float* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sGer(ctxt, m, n, h_alpha, x, incx, y, incy, AP, lda);
  HIPBLAS_CATCH("GER")
}

hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n, const double* alpha,
                            const double* x, int incx, const double* y, int incy,
                            double* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dGer(ctxt, m, n, h_alpha, x, incx, y, incy, AP, lda);
  HIPBLAS_CATCH("GER")
}

hipblasStatus_t hipblasCgerc(hipblasHandle_t handle, int m, int n, const hipblasComplex* alpha,
                            const hipblasComplex* x, int incx, const hipblasComplex* y, int incy,
                            hipblasComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
  }
  H4I::MKLShim::cGerc(ctxt, m, n, h_alpha, (const float _Complex*)x, incx, (const float _Complex*)y, incy,
              (float _Complex*)AP, lda);
  HIPBLAS_CATCH("GER")
}

hipblasStatus_t hipblasCgeru(hipblasHandle_t handle, int m, int n, const hipblasComplex* alpha,
                            const hipblasComplex* x, int incx, const hipblasComplex* y, int incy,
                            hipblasComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
  }
  H4I::MKLShim::cGeru(ctxt, m, n, h_alpha, (const float _Complex*)x, incx, (const float _Complex*)y, incy,
              (float _Complex*)AP, lda);
  HIPBLAS_CATCH("GER")
}

hipblasStatus_t hipblasZgerc(hipblasHandle_t handle, int m, int n, const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy,
                            hipblasDoubleComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zGerc(ctxt, m, n, h_alpha, (const double _Complex*)x, incx, (const double _Complex*)y, incy,
              (double _Complex*)AP, lda);
  HIPBLAS_CATCH("GER")
}

hipblasStatus_t hipblasZgeru(hipblasHandle_t handle, int m, int n, const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy,
                            hipblasDoubleComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
    m <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zGeru(ctxt, m, n, h_alpha, (const double _Complex*)x, incx, (const double _Complex*)y, incy,
              (double _Complex*)AP, lda);
  HIPBLAS_CATCH("GER")
}

// ger_batched
hipblasStatus_t hipblasSgerBatched(hipblasHandle_t    handle,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgerBatched(hipblasHandle_t     handle,
                                   int                 m,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeruBatched(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgercBatched(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasZgeruBatched(hipblasHandle_t                   handle,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgercBatched(hipblasHandle_t                   handle,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// ger_strided_batched
hipblasStatus_t hipblasSgerStridedBatched(hipblasHandle_t handle,
                                          int             m,
                                          int             n,
                                          const float*    alpha,
                                          const float*    x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          float*          A,
                                          int             lda,
                                          hipblasStride   strideA,
                                          int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle,
                                          int             m,
                                          int             n,
                                          const double*   alpha,
                                          const double*   x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const double*   y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          double*         A,
                                          int             lda,
                                          hipblasStride   strideA,
                                          int             batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeruStridedBatched(hipblasHandle_t       handle,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgercStridedBatched(hipblasHandle_t       handle,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeruStridedBatched(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgercStridedBatched(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : hbmv(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChbmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                            int n, int k, const hipblasComplex* alpha,
                            const hipblasComplex* AP, int lda,
                            const hipblasComplex* x, int incx,
                             const hipblasComplex* beta, hipblasComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    k <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cHbmv(ctxt, convert(uplo), n,k, h_alpha, (const float _Complex*)AP, lda, (const float _Complex*)x, incx, 
              h_beta, (float _Complex*)y, incy);
  HIPBLAS_CATCH("HBMV")
}

hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                            int n, int k, const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* AP, int lda,
                            const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    k <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zHbmv(ctxt, convert(uplo), n,k, h_alpha, (const double _Complex*)AP, lda, (const double _Complex*)x, incx, 
              h_beta, (double _Complex*)y, incy);
  HIPBLAS_CATCH("HBMV")
}

// Level-2 : hemv(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasComplex* alpha, const hipblasComplex* AP,
                            int lda, const hipblasComplex* x, int incx,
                            const hipblasComplex* beta, hipblasComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cHemv(ctxt, convert(uplo), n, h_alpha, (const float _Complex*)AP, lda,
              (const float _Complex*)x, incx, h_beta, (float _Complex*)y, incy);
  HIPBLAS_CATCH("HEMV")
}

hipblasStatus_t hipblasZhemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP,
                            int lda, const hipblasDoubleComplex* x, int incx,
                            const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
    n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zHemv(ctxt, convert(uplo), n, h_alpha, (const double _Complex*)AP, lda,
              (const double _Complex*)x, incx, h_beta, (double _Complex*)y, incy); 
  HIPBLAS_CATCH("HEMV")
}

// hemv_batched
hipblasStatus_t hipblasChemvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hemv_strided_batched
hipblasStatus_t hipblasChemvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stride_y,
                                           int                   batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stride_y,
                                           int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hbmv_batched
hipblasStatus_t hipblasChbmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount){
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhbmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount){
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hbmv_strided_batched
hipblasStatus_t hipblasChbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : her(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasCher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const float* alpha, const hipblasComplex* x, int incx,
                            hipblasComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
      n <= 0 || lda <= 0 ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::cHer(ctxt, convert(uplo), n, h_alpha, (const float _Complex*)x, incx, (float _Complex*)AP, lda);
  HIPBLAS_CATCH("HER")
}

hipblasStatus_t hipblasZher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const double* alpha, const hipblasDoubleComplex* x, int incx,
                            hipblasDoubleComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
      n <= 0 || lda <= 0 ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::zHer(ctxt, convert(uplo), n, h_alpha, (const double _Complex*)x, incx, (double _Complex*)AP, lda);
  HIPBLAS_CATCH("HER")
}

// her_batched
hipblasStatus_t hipblasCherBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const float*                alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her_strided_batched
hipblasStatus_t hipblasCherStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          hipblasStride         strideA,
                                          int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          hipblasStride               strideA,
                                          int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : her2(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasCher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasComplex* alpha, const hipblasComplex* x, int incx,
                            const hipblasComplex* y, int incy, hipblasComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
      n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
  }
  H4I::MKLShim::cHer2(ctxt, convert(uplo), n, h_alpha, (const float _Complex*)x, incx,
              (const float _Complex*)y, incy, (float _Complex*)AP, lda);
  HIPBLAS_CATCH("HER2")
}

hipblasStatus_t hipblasZher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx,
                            const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP, int lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
      n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zHer2(ctxt, convert(uplo), n, h_alpha, (const double _Complex*)x, incx,
              (const double _Complex*)y, incy, (double _Complex*)AP, lda);
  HIPBLAS_CATCH("HER2")
}
// her2_batched
hipblasStatus_t hipblasCher2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her2_strided_batched
hipblasStatus_t hipblasCher2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : hpmv(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasComplex* alpha, const hipblasComplex* AP,
                            const hipblasComplex* x, int incx, const hipblasComplex* beta,
                            hipblasComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
      n <= 0 ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cHpmv(ctxt, convert(uplo), n, h_alpha, (const float _Complex*)AP, (const float _Complex*)x, incx,
              h_beta, (float _Complex*)y, incy);
  HIPBLAS_CATCH("HPMV")
}

hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP,
                            const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta,
                            hipblasDoubleComplex* y, int incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
      n <= 0 ) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zHpmv(ctxt, convert(uplo), n, h_alpha, (const double _Complex*)AP, (const double _Complex*)x, incx,
              h_beta, (double _Complex*)y, incy);
  HIPBLAS_CATCH("HPMV")
}

// hpmv_batched
hipblasStatus_t hipblasChpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const AP[],
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const AP[],
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpmv_strided_batched
hipblasStatus_t hipblasChpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : hpr(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChpr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const float*          alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::cHpr(ctxt, convert(uplo), n, h_alpha, (const float _Complex*)x, incx, (float _Complex*)AP);
  HIPBLAS_CATCH("HPR")
}

hipblasStatus_t hipblasZhpr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const double*               alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::zHpr(ctxt, convert(uplo), n, h_alpha, (const double _Complex*)x, incx, (double _Complex*)AP);
  HIPBLAS_CATCH("HPR")
}

// hpr_batched
hipblasStatus_t hipblasChprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const float*                alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpr_strided_batched
hipblasStatus_t hipblasChprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       AP,
                                          hipblasStride         strideAP,
                                          int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       AP,
                                          hipblasStride               strideAP,
                                          int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : hpr2(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChpr2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
  }
  H4I::MKLShim::cHpr2(ctxt, convert(uplo), n, h_alpha, (const float _Complex*)x, incx,
                          (const float _Complex*)y, incy, (float _Complex*)AP);
  HIPBLAS_CATCH("HPR2")
}

hipblasStatus_t hipblasZhpr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zHpr2(ctxt, convert(uplo), n, h_alpha, (const double _Complex*)x, incx,
                          (const double _Complex*)y, incy, (double _Complex*)AP);
  HIPBLAS_CATCH("HPR2")
}

// hpr2_batched
hipblasStatus_t hipblasChpr2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       AP[],
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       AP[],
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpr2_strided_batched
hipblasStatus_t hipblasChpr2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       AP,
                                           hipblasStride         strideAP,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpr2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       AP,
                                           hipblasStride               strideAP,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : sbmv(supported datatypes : float and doule )
hipblasStatus_t hipblasSsbmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             int               k,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
      k <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sSbmv(ctxt, convert(uplo), n, k, h_alpha, A, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("SBMV")
}

hipblasStatus_t hipblasDsbmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             int               k,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
      k <= 0 || n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dSbmv(ctxt, convert(uplo), n, k, h_alpha, A, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("SBMV")
}

// sbmv_batched
hipblasStatus_t hipblasSsbmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float*             y[],
                                    int                incy,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    int                 k,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// sbmv_strided_batched
hipblasStatus_t hipblasSsbmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           int               k,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsbmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           int               k,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : spmv(supported datatypes : float and doule )
hipblasStatus_t hipblasSspmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      AP,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sSpmv(ctxt, convert(uplo), n, h_alpha, AP, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("SBMV")
}

hipblasStatus_t hipblasDspmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     AP,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dSpmv(ctxt, convert(uplo), n, h_alpha, AP, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("SBMV")
}

// spmv_batched
hipblasStatus_t hipblasSspmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const AP[],
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float*             y[],
                                    int                incy,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const AP[],
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spmv_strided_batched
hipblasStatus_t hipblasSspmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      AP,
                                           hipblasStride     strideAP,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     AP,
                                           hipblasStride     strideAP,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : spr(supported datatypes : float, double, float complex and doule complex )
hipblasStatus_t hipblasSspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sSpr(ctxt, convert(uplo), n, h_alpha, x, incx, AP);
  HIPBLAS_CATCH("SPR")
}

hipblasStatus_t hipblasDspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dSpr(ctxt, convert(uplo), n, h_alpha, x, incx, AP);
  HIPBLAS_CATCH("SPR")
}

hipblasStatus_t hipblasCspr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZspr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr_batched
hipblasStatus_t hipblasSsprBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   float* const       AP[],
                                   int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsprBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       AP[],
                                   int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr_strided_batched
hipblasStatus_t hipblasSsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          float*            AP,
                                          hipblasStride     strideAP,
                                          int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          double*           AP,
                                          hipblasStride     strideAP,
                                          int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       AP,
                                          hipblasStride         strideAP,
                                          int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       AP,
                                          hipblasStride               strideAP,
                                          int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : spr2(supported datatypes : float and double )
hipblasStatus_t hipblasSspr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      x,
                             int               incx,
                             const float*      y,
                             int               incy,
                             float*            AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sSpr2(ctxt, convert(uplo), n, h_alpha, x, incx, y, incy, AP);
  HIPBLAS_CATCH("SPR2")
}

hipblasStatus_t hipblasDspr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     x,
                             int               incx,
                             const double*     y,
                             int               incy,
                             double*           AP){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
      n <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dSpr2(ctxt, convert(uplo), n, h_alpha, x, incx, y, incy, AP);
  HIPBLAS_CATCH("SPR2")
}

// spr2_batched
hipblasStatus_t hipblasSspr2Batched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    const float* const y[],
                                    int                incy,
                                    float* const       AP[],
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspr2Batched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    const double* const y[],
                                    int                 incy,
                                    double* const       AP[],
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr2_strided_batched
hipblasStatus_t hipblasSspr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           float*            AP,
                                           hipblasStride     strideAP,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           double*           AP,
                                           hipblasStride     strideAP,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : symv(supported datatypes : float and double )
hipblasStatus_t hipblasSsymv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr || beta == nullptr ||
      n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sSymv(ctxt, convert(uplo), n, h_alpha, A, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("SYMV")
}

hipblasStatus_t hipblasDsymv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr || beta == nullptr ||
      n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dSymv(ctxt, convert(uplo), n, h_alpha, A, lda, x, incx, h_beta, y, incy);
  HIPBLAS_CATCH("SYMV")
}

hipblasStatus_t hipblasCsymv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* beta,
                             hipblasComplex*       y,
                             int                   incy) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       y,
                             int                         incy) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symv_batched
hipblasStatus_t hipblasSsymvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float*             y[],
                                    int                incy,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex*             y[],
                                    int                         incy,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex*             y[],
                                    int                               incy,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symv_strided_batched
hipblasStatus_t hipblasSsymvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasDsymvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : syr(supported datatypes : float and double )
hipblasStatus_t hipblasSsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            A,
                            int               lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr ||
      n <= 0 || incx <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sSyr(ctxt, convert(uplo), n, h_alpha, x, incx, A, lda);
  HIPBLAS_CATCH("SYR")
}

hipblasStatus_t hipblasDsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           A,
                            int               lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr ||
      n <= 0 || incx <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dSyr(ctxt, convert(uplo), n, h_alpha, x, incx, A, lda);
  HIPBLAS_CATCH("SYR")
}

hipblasStatus_t hipblasCsyr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       A,
                            int                   lda) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       A,
                            int                         lda) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr_batched
hipblasStatus_t hipblasSsyrBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasCsyrBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr_strided_batched
hipblasStatus_t hipblasSsyrStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          float*            A,
                                          int               lda,
                                          hipblasStride     strideA,
                                          int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          double*           A,
                                          int               lda,
                                          hipblasStride     strideA,
                                          int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          hipblasStride         strideA,
                                          int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          hipblasStride               strideA,
                                          int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : syr2(supported datatypes : float and double )
hipblasStatus_t hipblasSsyr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      x,
                             int               incx,
                             const float*      y,
                             int               incy,
                             float*            A,
                             int               lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr ||
      n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sSyr2(ctxt, convert(uplo), n, h_alpha, x, incx, y, incy, A, lda);
  HIPBLAS_CATCH("SYR2")
}

hipblasStatus_t hipblasDsyr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     x,
                             int               incx,
                             const double*     y,
                             int               incy,
                             double*           A,
                             int               lda){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr ||
      n <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dSyr2(ctxt, convert(uplo), n, h_alpha, x, incx, y, incy, A, lda);
  HIPBLAS_CATCH("SYR2")
}
hipblasStatus_t hipblasCsyr2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       A,
                             int                   lda) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       A,
                             int                         lda) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2_batched
hipblasStatus_t hipblasSsyr2Batched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    const float* const y[],
                                    int                incy,
                                    float* const       A[],
                                    int                lda,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2Batched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    const double* const y[],
                                    int                 incy,
                                    double* const       A[],
                                    int                 lda,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2_strided_batched
hipblasStatus_t hipblasSsyr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           float*            A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           double*           A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : tbmv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sTbmv(ctxt, convert(uplo), convert(transA), convert(diag), m, k, A, lda, x, incx);
  HIPBLAS_CATCH("TBMV")
}

hipblasStatus_t hipblasDtbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dTbmv(ctxt, convert(uplo), convert(transA), convert(diag), m, k, A, lda, x, incx);
  HIPBLAS_CATCH("TBMV")
}

hipblasStatus_t hipblasCtbmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cTbmv(ctxt, convert(uplo), convert(transA), convert(diag), m, k, 
                          (const float _Complex*)A, lda, (float _Complex*)x, incx);
  HIPBLAS_CATCH("TBMV")
}

hipblasStatus_t hipblasZtbmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zTbmv(ctxt, convert(uplo), convert(transA), convert(diag), m, k, 
                          (const double _Complex*)A, lda, (double _Complex*)x, incx);
  HIPBLAS_CATCH("TBMV")
}

// tbmv_batched
hipblasStatus_t hipblasStbmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tbmv_strided_batched
hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           int                   batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : tbsv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStbsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      n <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sTbsv(ctxt, convert(uplo), convert(transA), convert(diag), n, k, A, lda, x, incx);
  HIPBLAS_CATCH("TBSV")
}

hipblasStatus_t hipblasDtbsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      n <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dTbsv(ctxt, convert(uplo), convert(transA), convert(diag), n, k, A, lda, x, incx);
  HIPBLAS_CATCH("TBSV")
}


hipblasStatus_t hipblasCtbsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      n <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cTbsv(ctxt, convert(uplo), convert(transA), convert(diag), n, k, (const float _Complex*)A, lda, (float _Complex*)x, incx);
  HIPBLAS_CATCH("TBSV")
}


hipblasStatus_t hipblasZtbsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      n <= 0 || k <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zTbsv(ctxt, convert(uplo), convert(transA), convert(diag), n, k, (const double _Complex*)A, lda, (double _Complex*)x, incx);
  HIPBLAS_CATCH("TBSV")
}


// tbsv_batched
hipblasStatus_t hipblasStbsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tbsv_strided_batched
hipblasStatus_t hipblasStbsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : tpmv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sTpmv(ctxt, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
  HIPBLAS_CATCH("TPMV")
}

hipblasStatus_t hipblasDtpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dTpmv(ctxt, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
  HIPBLAS_CATCH("TPMV")
}

hipblasStatus_t hipblasCtpmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cTpmv(ctxt, convert(uplo), convert(transA), convert(diag), m, 
              (const float _Complex*)AP, (float _Complex*)x, incx);
  HIPBLAS_CATCH("TPMV")
}

hipblasStatus_t hipblasZtpmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zTpmv(ctxt, convert(uplo), convert(transA), convert(diag), m, 
              (const double _Complex*)AP, (double _Complex*)x, incx);
  HIPBLAS_CATCH("TPMV")
}

// tpmv_batched
hipblasStatus_t hipblasStpmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tpmv_strided_batched
hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : tpsv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sTpsv(ctxt, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
  HIPBLAS_CATCH("TPSV")
}

hipblasStatus_t hipblasDtpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dTpsv(ctxt, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
  HIPBLAS_CATCH("TPSV")
}

hipblasStatus_t hipblasCtpsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cTpsv(ctxt, convert(uplo), convert(transA), convert(diag), m,
              (const float _Complex*) AP, (float _Complex*)x, incx);
  HIPBLAS_CATCH("TPSV")
}
hipblasStatus_t hipblasZtpsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx){
  HIPBLAS_TRY
  if (handle == nullptr || AP == nullptr || x == nullptr ||
      m <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zTpsv(ctxt, convert(uplo), convert(transA), convert(diag), m,
              (const double _Complex*) AP, (double _Complex*)x, incx);
  HIPBLAS_CATCH("TPSV")
}

// tpsv_batched
hipblasStatus_t hipblasStpsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tpsv_strided_batched
hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : trmv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sTrmv(ctxt, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
  HIPBLAS_CATCH("TRMV")
}

hipblasStatus_t hipblasDtrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dTrmv(ctxt, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
  HIPBLAS_CATCH("TRMV")
}

hipblasStatus_t hipblasCtrmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cTrmv(ctxt, convert(uplo), convert(transA), convert(diag), m,
              (const float _Complex*)A, lda, (float _Complex*)x, incx);
  HIPBLAS_CATCH("TRMV")
}

hipblasStatus_t hipblasZtrmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 || lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zTrmv(ctxt, convert(uplo), convert(transA), convert(diag), m,
              (const double _Complex*)A, lda, (double _Complex*)x, incx);
  HIPBLAS_CATCH("TRMV")
}

// trmv_batched
hipblasStatus_t hipblasStrmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasCtrmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trmv_strided_batched
hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : trsv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 ||lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::sTrsv(ctxt, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
  HIPBLAS_CATCH("TRSV")
}

hipblasStatus_t hipblasDtrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 ||lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::dTrsv(ctxt, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
  HIPBLAS_CATCH("TRSV")
}

hipblasStatus_t hipblasCtrsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 ||lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::cTrsv(ctxt, convert(uplo), convert(transA), convert(diag), m,
              (const float _Complex*)A, lda, (float _Complex*)x, incx);
  HIPBLAS_CATCH("TRSV")
}

hipblasStatus_t hipblasZtrsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx){
  HIPBLAS_TRY
  if (handle == nullptr || A == nullptr || x == nullptr ||
      m <= 0 ||lda <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::zTrsv(ctxt, convert(uplo), convert(transA), convert(diag), m,
              (const double _Complex*)A, lda, (double _Complex*)x, incx);
  HIPBLAS_CATCH("TRSV")
}

// trsv_batched
hipblasStatus_t hipblasStrsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trsv_strided_batched
hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// Level-3 : herk(supported datatypes : float complex and double complex )
hipblasStatus_t hipblasCherk(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             int                   n,
                             int                   k,
                             const float*          alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const float*          beta,
                             hipblasComplex*       C,
                             int                   ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::cHerk(ctxt, convert(uplo), convert(transA), n, k,
              h_alpha, (const float _Complex*)A, lda, h_beta, (float _Complex*)C, ldc);
  HIPBLAS_CATCH("HERK")
}

hipblasStatus_t hipblasZherk(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             int                         n,
                             int                         k,
                             const double*               alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const double*               beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::zHerk(ctxt, convert(uplo), convert(transA), n, k,
              h_alpha, (const double _Complex*)A, lda, h_beta, (double _Complex*)C, ldc);
  HIPBLAS_CATCH("HERK")
}

// herk_batched
hipblasStatus_t hipblasCherkBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int                         n,
                                    int                         k,
                                    const float*                alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const float*                beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    int                               n,
                                    int                               k,
                                    const double*                     alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const double*                     beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// herk_strided_batched
hipblasStatus_t hipblasCherkStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const float*          alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const double*               alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : herkx(supported datatypes : float complex , double complex )
hipblasStatus_t hipblasCherkx(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              int                   n,
                              int                   k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A,
                              int                   lda,
                              const hipblasComplex* B,
                              int                   ldb,
                              const float*          beta,
                              hipblasComplex*       C,
                              int                   ldc) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkx(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasOperation_t          transA,
                              int                         n,
                              int                         k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              const hipblasDoubleComplex* B,
                              int                         ldb,
                              const double*               beta,
                              hipblasDoubleComplex*       C,
                              int                         ldc) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// herkx_batched
hipblasStatus_t hipblasCherkxBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasComplex*       alpha,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     const hipblasComplex* const B[],
                                     int                         ldb,
                                     const float*                beta,
                                     hipblasComplex* const       C[],
                                     int                         ldc,
                                     int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkxBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasOperation_t                transA,
                                     int                               n,
                                     int                               k,
                                     const hipblasDoubleComplex*       alpha,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     const hipblasDoubleComplex* const B[],
                                     int                               ldb,
                                     const double*                     beta,
                                     hipblasDoubleComplex* const       C[],
                                     int                               ldc,
                                     int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// herkx_strided_batched
hipblasStatus_t hipblasCherkxStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasStride         strideB,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            hipblasStride         strideC,
                                            int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkxStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasStride               strideB,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            hipblasStride               strideC,
                                            int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-2 : her2k(supported datatypes : float complex and double complex )
hipblasStatus_t hipblasCher2k(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              int                   n,
                              int                   k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A,
                              int                   lda,
                              const hipblasComplex* B,
                              int                   ldb,
                              const float*          beta,
                              hipblasComplex*       C,
                              int                   ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  float h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::cHer2k(ctxt, convert(uplo), convert(transA), n, k, h_alpha,
              (const float _Complex*)A, lda, (const float _Complex*)B, ldb, h_beta,
              (float _Complex*)C, ldc);
  HIPBLAS_CATCH("HER2K")
}

hipblasStatus_t hipblasZher2k(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasOperation_t          transA,
                              int                         n,
                              int                         k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              const hipblasDoubleComplex* B,
                              int                         ldb,
                              const double*               beta,
                              hipblasDoubleComplex*       C,
                              int                         ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  double h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::zHer2k(ctxt, convert(uplo), convert(transA), n, k, h_alpha,
              (const double _Complex*)A, lda, (const double _Complex*)B, ldb, h_beta,
              (double _Complex*)C, ldc);
  HIPBLAS_CATCH("HER2K")
}

// her2k_batched
hipblasStatus_t hipblasCher2kBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasComplex*       alpha,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     const hipblasComplex* const B[],
                                     int                         ldb,
                                     const float*                beta,
                                     hipblasComplex* const       C[],
                                     int                         ldc,
                                     int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2kBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasOperation_t                transA,
                                     int                               n,
                                     int                               k,
                                     const hipblasDoubleComplex*       alpha,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     const hipblasDoubleComplex* const B[],
                                     int                               ldb,
                                     const double*                     beta,
                                     hipblasDoubleComplex* const       C[],
                                     int                               ldc,
                                     int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her2k_strided_batched
hipblasStatus_t hipblasCher2kStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasStride         strideB,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            hipblasStride         strideC,
                                            int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2kStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasStride               strideB,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            hipblasStride               strideC,
                                            int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : symm(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasSsymm(hipblasHandle_t   handle,
                             hipblasSideMode_t side,
                             hipblasFillMode_t uplo,
                             int               m,
                             int               n,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      B,
                             int               ldb,
                             const float*      beta,
                             float*            C,
                             int               ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sSymm(ctxt, convert(side), convert(uplo), m, n, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
  HIPBLAS_CATCH("SYMM")
}

hipblasStatus_t hipblasDsymm(hipblasHandle_t   handle,
                             hipblasSideMode_t side,
                             hipblasFillMode_t uplo,
                             int               m,
                             int               n,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     B,
                             int               ldb,
                             const double*     beta,
                             double*           C,
                             int               ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dSymm(ctxt, convert(side), convert(uplo), m, n, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
  HIPBLAS_CATCH("SYMM")
}

hipblasStatus_t hipblasCsymm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symm_batched
hipblasStatus_t hipblasSsymmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const B[],
                                    int                ldb,
                                    const float*       beta,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymmBatched(hipblasHandle_t     handle,
                                    hipblasSideMode_t   side,
                                    hipblasFillMode_t   uplo,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const B[],
                                    int                 ldb,
                                    const double*       beta,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symm_strided_batched
hipblasStatus_t hipblasSsymmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           hipblasFillMode_t uplo,
                                           int               m,
                                           int               n,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const float*      B,
                                           int               ldb,
                                           hipblasStride     strideB,
                                           const float*      beta,
                                           float*            C,
                                           int               ldc,
                                           hipblasStride     strideC,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           hipblasFillMode_t uplo,
                                           int               m,
                                           int               n,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const double*     B,
                                           int               ldb,
                                           hipblasStride     strideB,
                                           const double*     beta,
                                           double*           C,
                                           int               ldc,
                                           hipblasStride     strideC,
                                           int               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : syrk(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasSsyrk(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             const float*       beta,
                             float*             C,
                             int                ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sSyrk(ctxt, convert(uplo), convert(transA), n, k, h_alpha, A, lda, h_beta, C, ldc);
  HIPBLAS_CATCH("SYRK")
}

hipblasStatus_t hipblasDsyrk(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             const double*      beta,
                             double*            C,
                             int                ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dSyrk(ctxt, convert(uplo), convert(transA), n, k, h_alpha, A, lda, h_beta, C, ldc);
  HIPBLAS_CATCH("SYRK")
}

hipblasStatus_t hipblasCsyrk(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cSyrk(ctxt, convert(uplo), convert(transA), n, k, h_alpha, (const float _Complex*)A, lda,
              h_beta, (float _Complex*)C, ldc);
  HIPBLAS_CATCH("SYRK")
}
hipblasStatus_t hipblasZsyrk(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zSyrk(ctxt, convert(uplo), convert(transA), n, k, h_alpha, (const double _Complex*)A, lda,
              h_beta, (double _Complex*)C, ldc);
  HIPBLAS_CATCH("SYRK")
}

// syrk_batched
hipblasStatus_t hipblasSsyrkBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float*       beta,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrkBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    int                 n,
                                    int                 k,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double*       beta,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syrk_strided_batched
hipblasStatus_t hipblasSsyrkStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           hipblasStride      strideC,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrkStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           hipblasStride      strideC,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : syr2k(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasSsyr2k(hipblasHandle_t    handle,
                              hipblasFillMode_t  uplo,
                              hipblasOperation_t transA,
                              int                n,
                              int                k,
                              const float*       alpha,
                              const float*       A,
                              int                lda,
                              const float*       B,
                              int                ldb,
                              const float*       beta,
                              float*             C,
                              int                ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sSyr2k(ctxt, convert(uplo), convert(transA), n, k, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
  HIPBLAS_CATCH("SYR2K")
}

hipblasStatus_t hipblasDsyr2k(hipblasHandle_t    handle,
                              hipblasFillMode_t  uplo,
                              hipblasOperation_t transA,
                              int                n,
                              int                k,
                              const double*      alpha,
                              const double*      A,
                              int                lda,
                              const double*      B,
                              int                ldb,
                              const double*      beta,
                              double*            C,
                              int                ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dSyr2k(ctxt, convert(uplo), convert(transA), n, k, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
  HIPBLAS_CATCH("SYR2K")
}

hipblasStatus_t hipblasCsyr2k(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              int                   n,
                              int                   k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A,
                              int                   lda,
                              const hipblasComplex* B,
                              int                   ldb,
                              const hipblasComplex* beta,
                              hipblasComplex*       C,
                              int                   ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cSyr2k(ctxt, convert(uplo), convert(transA), n, k, h_alpha,
                (const float _Complex*)A, lda, (const float _Complex*)B, ldb, h_beta, (float _Complex*)C, ldc);
  HIPBLAS_CATCH("SYR2K")
}

hipblasStatus_t hipblasZsyr2k(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasOperation_t          transA,
                              int                         n,
                              int                         k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              const hipblasDoubleComplex* B,
                              int                         ldb,
                              const hipblasDoubleComplex* beta,
                              hipblasDoubleComplex*       C,
                              int                         ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zSyr2k(ctxt, convert(uplo), convert(transA), n, k, h_alpha,
                (const double _Complex*)A, lda, (const double _Complex*)B, ldb, h_beta, (double _Complex*)C, ldc);
  HIPBLAS_CATCH("SYR2K")
}

// syr2k_batched
hipblasStatus_t hipblasSsyr2kBatched(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const float*       alpha,
                                     const float* const A[],
                                     int                lda,
                                     const float* const B[],
                                     int                ldb,
                                     const float*       beta,
                                     float* const       C[],
                                     int                ldc,
                                     int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2kBatched(hipblasHandle_t     handle,
                                     hipblasFillMode_t   uplo,
                                     hipblasOperation_t  transA,
                                     int                 n,
                                     int                 k,
                                     const double*       alpha,
                                     const double* const A[],
                                     int                 lda,
                                     const double* const B[],
                                     int                 ldb,
                                     const double*       beta,
                                     double* const       C[],
                                     int                 ldc,
                                     int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2kBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasComplex*       alpha,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     const hipblasComplex* const B[],
                                     int                         ldb,
                                     const hipblasComplex*       beta,
                                     hipblasComplex* const       C[],
                                     int                         ldc,
                                     int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2kBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasOperation_t                transA,
                                     int                               n,
                                     int                               k,
                                     const hipblasDoubleComplex*       alpha,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     const hipblasDoubleComplex* const B[],
                                     int                               ldb,
                                     const hipblasDoubleComplex*       beta,
                                     hipblasDoubleComplex* const       C[],
                                     int                               ldc,
                                     int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2k_strided_batched
hipblasStatus_t hipblasSsyr2kStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            hipblasStride      strideA,
                                            const float*       B,
                                            int                ldb,
                                            hipblasStride      strideB,
                                            const float*       beta,
                                            float*             C,
                                            int                ldc,
                                            hipblasStride      strideC,
                                            int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2kStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            hipblasStride      strideA,
                                            const double*      B,
                                            int                ldb,
                                            hipblasStride      strideB,
                                            const double*      beta,
                                            double*            C,
                                            int                ldc,
                                            hipblasStride      strideC,
                                            int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2kStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasStride         strideB,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            hipblasStride         strideC,
                                            int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2kStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasStride               strideB,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            hipblasStride               strideC,
                                            int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : hemm(supported datatypes : float complex and double complex )
hipblasStatus_t hipblasChemm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cHemm(ctxt, convert(side), convert(uplo), n, k, h_alpha, (const float _Complex*)A, lda, 
              (const float _Complex*)B, ldb, h_beta, (float _Complex*)C, ldc);
  HIPBLAS_CATCH("HEMM")
}

hipblasStatus_t hipblasZhemm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zHemm(ctxt, convert(side), convert(uplo), n, k, h_alpha, (const double _Complex*)A, lda, 
              (const double _Complex*)B, ldb, h_beta, (double _Complex*)C, ldc);
  HIPBLAS_CATCH("HEMM")
}

// hemm_batched
hipblasStatus_t hipblasChemmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hemm_strided_batched
hipblasStatus_t hipblasChemmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : trmm(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrmm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             float*             B,
                             int                ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sTrmm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
  HIPBLAS_CATCH("TRMM")
}

hipblasStatus_t hipblasDtrmm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             double*            B,
                             int                ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dTrmm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
  HIPBLAS_CATCH("TRMM")
}

hipblasStatus_t hipblasCtrmm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       B,
                             int                   ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
  }
  H4I::MKLShim::cTrmm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha,
              (const float _Complex*)A, lda, (float _Complex*)B, ldb);
  HIPBLAS_CATCH("TRMM")
}

hipblasStatus_t hipblasZtrmm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       B,
                             int                         ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zTrmm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha,
              (const double _Complex*)A, lda, (double _Complex*)B, ldb);
  HIPBLAS_CATCH("TRMM")
}

// trmm_batched
hipblasStatus_t hipblasStrmmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    float* const       B[],
                                    int                ldb,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmmBatched(hipblasHandle_t     handle,
                                    hipblasSideMode_t   side,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       B[],
                                    int                 ldb,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       B[],
                                    int                         ldb,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       B[],
                                    int                               ldb,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trmm_strided_batched
hipblasStatus_t hipblasStrmmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : trsm(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrsm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const float*       alpha,
                             float*             A,
                             int                lda,
                             float*             B,
                             int                ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
  }
  H4I::MKLShim::sTrsm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
  HIPBLAS_CATCH("TRSM")
}

hipblasStatus_t hipblasDtrsm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const double*      alpha,
                             double*            A,
                             int                lda,
                             double*            B,
                             int                ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
  }
  H4I::MKLShim::dTrsm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
  HIPBLAS_CATCH("TRSM")
}

hipblasStatus_t hipblasCtrsm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             hipblasComplex*       A,
                             int                   lda,
                             hipblasComplex*       B,
                             int                   ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
  }
  H4I::MKLShim::cTrsm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n,
              h_alpha, (const float _Complex*)A, lda, (float _Complex*)B, ldb);
  HIPBLAS_CATCH("TRSM")
}

hipblasStatus_t hipblasZtrsm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             hipblasDoubleComplex*       A,
                             int                         lda,
                             hipblasDoubleComplex*       B,
                             int                         ldb){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
      m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
  }
  H4I::MKLShim::zTrsm(ctxt, convert(side), convert(uplo), convert(transA), convert(diag), m, n,
              h_alpha, (const double _Complex*)A, lda, (double _Complex*)B, ldb);
  HIPBLAS_CATCH("TRSM")
}

// trsm_batched
hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    float* const       A[],
                                    int                lda,
                                    float*             B[],
                                    int                ldb,
                                    int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    double* const      A[],
                                    int                lda,
                                    double*            B[],
                                    int                ldb,
                                    int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsmBatched(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const A[],
                                    int                   lda,
                                    hipblasComplex*       B[],
                                    int                   ldb,
                                    int                   batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const A[],
                                    int                         lda,
                                    hipblasDoubleComplex*       B[],
                                    int                         ldb,
                                    int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trsm_strided_batched
hipblasStatus_t hipblasStrsmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           float*             A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           double*            A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           int                   batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           int                         batch_count) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Level-3 : gemm(supported datatypes : half, float , double , float complex and double complex )
hipblasStatus_t hipblasHgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const hipblasHalf* alpha,
                             const hipblasHalf* A,
                             int                lda,
                             const hipblasHalf* B,
                             int                ldb,
                             const hipblasHalf* beta,
                             hipblasHalf*       C,
                             int                ldc){
 // How to handle half??
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasSgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             const float*       B,
                             int                ldb,
                             const float*       beta,
                             float*             C,
                             int                ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
  } else {
      h_alpha = *((float*)alpha);
      h_beta = *((float*)beta);
  }

  H4I::MKLShim::sGemm(ctxt, convert(transa), convert(transb), m, n, k,
              h_alpha, A, lda, B, ldb, h_beta, C, ldc);
  HIPBLAS_CATCH("GEMM")
}

hipblasStatus_t hipblasDgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             const double*      B,
                             int                ldb,
                             const double*      beta,
                             double*            C,
                             int                ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
  } else {
      h_alpha = *((double*)alpha);
      h_beta = *((double*)beta);
  }

  H4I::MKLShim::dGemm(ctxt, convert(transa), convert(transb), m, n, k,
              h_alpha, A, lda, B, ldb, h_beta, C, ldc);
  HIPBLAS_CATCH("GEMM")
}

hipblasStatus_t hipblasCgemm(hipblasHandle_t       handle,
                             hipblasOperation_t    transa,
                             hipblasOperation_t    transb,
                             int                   m,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  float _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((float _Complex*)alpha);
      h_beta = *((float _Complex*)beta);
  }

  H4I::MKLShim::cGemm(ctxt, convert(transa), convert(transb), m, n, k,
              h_alpha, (const float _Complex*)A, lda, (const float _Complex*)B, ldb,
              h_beta, (float _Complex*)C, ldc);
  HIPBLAS_CATCH("GEMM")
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t             handle,
                             hipblasOperation_t          transa,
                             hipblasOperation_t          transb,
                             int                         m,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc){
  HIPBLAS_TRY
  if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
  hipblasPointerMode_t pointerMode;
  hipblasGetPointerMode(handle, &pointerMode);
  bool is_dev_ptr = (pointerMode == HIPBLAS_POINTER_MODE_DEVICE);

  double _Complex h_alpha, h_beta;
  if (is_dev_ptr) {
      hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
      hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
  } else {
      h_alpha = *((double _Complex*)alpha);
      h_beta = *((double _Complex*)beta);
  }

  H4I::MKLShim::zGemm(ctxt, convert(transa), convert(transb), m, n, k,
              h_alpha, (const double _Complex*)A, lda, (const double _Complex*)B, ldb,
              h_beta, (double _Complex*)C, ldc);
  HIPBLAS_CATCH("GEMM")
}

// gemm_batched
hipblasStatus_t hipblasHgemmBatched(hipblasHandle_t          handle,
                                    hipblasOperation_t       transa,
                                    hipblasOperation_t       transb,
                                    int                      m,
                                    int                      n,
                                    int                      k,
                                    const hipblasHalf*       alpha,
                                    const hipblasHalf* const A[],
                                    int                      lda,
                                    const hipblasHalf* const B[],
                                    int                      ldb,
                                    const hipblasHalf*       beta,
                                    hipblasHalf* const       C[],
                                    int                      ldc,
                                    int                      batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const B[],
                                    int                ldb,
                                    const float*       beta,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t     handle,
                                    hipblasOperation_t  transa,
                                    hipblasOperation_t  transb,
                                    int                 m,
                                    int                 n,
                                    int                 k,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const B[],
                                    int                 ldb,
                                    const double*       beta,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          transa,
                                    hipblasOperation_t          transb,
                                    int                         m,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemmBatched(hipblasHandle_t                   handle,
                                    hipblasOperation_t                transa,
                                    hipblasOperation_t                transb,
                                    int                               m,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gemm_strided_batched
hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const hipblasHalf* alpha,
                                           const hipblasHalf* A,
                                           int                lda,
                                           long long          bsa,
                                           const hipblasHalf* B,
                                           int                ldb,
                                           long long          bsb,
                                           const hipblasHalf* beta,
                                           hipblasHalf*       C,
                                           int                ldc,
                                           long long          bsc,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           long long          bsa,
                                           const float*       B,
                                           int                ldb,
                                           long long          bsb,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           long long          bsc,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           long long          bsa,
                                           const double*      B,
                                           int                ldb,
                                           long long          bsb,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           long long          bsc,
                                           int                batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgemmStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    transa,
                                           hipblasOperation_t    transb,
                                           int                   m,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           long long             bsa,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           long long             bsb,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           long long             bsc,
                                           int                   batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemmStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          transa,
                                           hipblasOperation_t          transb,
                                           int                         m,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           long long                   bsa,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           long long                   bsb,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           long long                   bsc,
                                           int                         batchCount) {
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}