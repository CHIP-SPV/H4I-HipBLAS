// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklblas.h"
#include "h4i/hipblas/impl/util.h"

// Level-1 : amax (supported datatypes : float, double, complex float, complex double)
// Generic amax which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result) {
  hipblasStatus_t ret = HIPBLAS_STATUS_SUCCESS;
  if (handle == nullptr) {
    return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
  }

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  try {
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;
    hipError_t hip_status;
    if (!is_result_dev_ptr)
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    H4I::MKLShim::onemklSamax(ctxt, n, x, incx, (int64_t*)result);

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;
        hip_status = hipFree(&dev_results);
    }
  }
  catch(std::exception const& e) {
    std::cerr << "MAX exception: " << e.what() << std::endl;
    return HIPBLAS_STATUS_EXECUTION_FAILED;
  }
  return HIPBLAS_STATUS_SUCCESS;
}