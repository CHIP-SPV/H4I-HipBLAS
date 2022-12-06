// Copyright 2021-2022 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hip/hip_interop.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hipblas/impl/Operation.h"

hipblasStatus_t
hipblasCreate(hipblasHandle_t* handle)
{
    if(handle != nullptr)
    {
        *handle = H4I::MKLShim::Create();
    }
    return (*handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t
hipblasDestroy(hipblasHandle_t handle)
{
    if(handle != nullptr)
    {
        H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
        H4I::MKLShim::Destroy(ctxt);
    }
    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t
hipblasSetStream(hipblasHandle_t handle, hipStream_t stream)
{
    if(handle != nullptr)
    {
        H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

        // Obtain the underlying CHIP-SPV handles.
        // Note this code uses a CHIP-SPV extension to the HIP API.
        // See CHIP-SPV documentation for its use.
        // Both Level Zero and OpenCL backends currently require us
        // to pass nHandles = 4, and provide space for at least 4 handles.
        // TODO is there a way to query this info at runtime?
        int nHandles = H4I::MKLShim::nHandles;
        std::array<uintptr_t, H4I::MKLShim::nHandles> nativeHandles;
        hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream),
                nativeHandles.data(), &nHandles);

        H4I::MKLShim::SetStream(ctxt, nativeHandles);
    }
    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

