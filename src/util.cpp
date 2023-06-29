// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include <atomic>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hip/hip_interop.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hipblas/impl/util.h"

// As of now keeping it as global variable, in future
// it will be part of Context so that different blas context can have it's own pointer mode
std::atomic<int> POINTER_MODE;

hipblasStatus_t
hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode)
{
    if (handle == nullptr) {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    if (mode != HIPBLAS_POINTER_MODE_HOST && mode != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    POINTER_MODE.store((int)mode);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode)
{
    if (mode == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    *mode = static_cast<hipblasPointerMode_t>(POINTER_MODE.load());
    return HIPBLAS_STATUS_SUCCESS;
}

struct HipHandles
{
    H4I::MKLShim::NativeHandleArray handles;
    int nHandles = handles.size();

    void GetNativeHandles(hipStream_t stream)
    {    
        // Note this code uses a chipStar extension to the HIP API.
        // See chipStar documentation for its use.
        // Both Level Zero and OpenCL backends currently require us
        // to pass nHandles = 4, and provide space for at least 4 handles.
        // TODO is there a way to query this info at runtime?
        hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream),
            handles.data(), &nHandles);
    }
};

hipblasStatus_t
hipblasCreate(hipblasHandle_t* handle)
{
    if(handle != nullptr)
    {
        // Determine the backend we're using.
        auto backend = H4I::MKLShim::ToBackend(hipGetBackendName());

        // Obtain the native backend handles.
        HipHandles hipHandles;
        hipHandles.GetNativeHandles(nullptr);
        *handle = H4I::MKLShim::Create(hipHandles.handles, backend);
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

        // Obtain the underlying chipStar handles.
        HipHandles hipHandles;
        hipHandles.GetNativeHandles(stream);

        H4I::MKLShim::SetStream(ctxt, hipHandles.handles);
    }
    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t
hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpy(y, x, elemSize * n, hipMemcpyHostToDevice);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}

hipblasStatus_t
hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpy(y, x, elemSize * n, hipMemcpyDeviceToHost);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}

hipblasStatus_t
hipblasSetVectorAsync(int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpyAsync(y, x, elemSize * n, hipMemcpyHostToDevice, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}

hipblasStatus_t
hipblasGetVectorAsync(int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpyAsync(y, x, elemSize * n, hipMemcpyDeviceToHost, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}

hipblasStatus_t
hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous h2d copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpy(B, A, no_of_bytes, hipMemcpyHostToDevice);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}

hipblasStatus_t
hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous d2h copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpy(B, A, no_of_bytes, hipMemcpyDeviceToHost);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}

hipblasStatus_t
hipblasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous h2d copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpyAsync(B, A, no_of_bytes, hipMemcpyHostToDevice, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}

hipblasStatus_t
hipblasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous d2h copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpyAsync(B, A, no_of_bytes, hipMemcpyDeviceToHost, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}
