// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hipblas/impl/util.h"


hipblasStatus_t
hipblasSgemm(hipblasHandle_t handle,
                hipblasOperation_t transa,
                hipblasOperation_t transb,
                int m,
                int n,
                int k,
                const float* alpha,
                const float* A,
                int ldA,
                const float* B,
                int ldB,
                const float* beta,
                float* C,
                int ldC)
{
    hipblasStatus_t ret = HIPBLAS_STATUS_SUCCESS;
    if(handle != nullptr)
    {
        auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

        try
        {
            H4I::MKLShim::SGEMM(ctxt,
                                ToMKLShimOp(transa),
                                ToMKLShimOp(transb),
                                m,
                                n,
                                k,
                                alpha,
                                A,
                                ldA,
                                B,
                                ldB,
                                beta,
                                C,
                                ldC);
        }
        catch(std::exception const& e)
        {
            std::cerr << "SGEMM exception: " << e.what() << std::endl;
            ret = HIPBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    else
    {
        ret = HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    return ret;
}

