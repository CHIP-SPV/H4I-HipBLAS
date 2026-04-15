// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hip/hip_interop.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hipblas/impl/util.h"



// As of now keeping it as global variable, in future
// it will be part of Context so that different blas context can have it's own pointer mode
std::atomic<int> POINTER_MODE;

namespace {
// MKLShim::Context does not carry the user's hipStream_t, but consumers
// (e.g. libCEED's hip-ref backend) expect hipblasGetStream to return whatever
// was last passed to hipblasSetStream. Track it in a side-map here rather than
// adding a field to MKLShim::Context (avoids an ABI break in MKLShim).
std::mutex g_handle_stream_mu;
std::unordered_map<hipblasHandle_t, hipStream_t> g_handle_stream;
}

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

hipblasStatus_t
hipblasCreate(hipblasHandle_t* handle)
{
    if(handle != nullptr)
    {
        #ifdef hipGetBackendName
        std::cerr << "Error: The hipGetBackendName API is deprecated. Please update your H4I-MKLShim to use the latest API." << std::endl;
        return HIPBLAS_STATUS_INTERNAL_ERROR;
        #endif

        int nHandles;
        hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);

        unsigned long handles[nHandles];
        hipGetBackendNativeHandles((uintptr_t)NULL, handles, 0);
        *handle = H4I::MKLShim::Create(handles, nHandles);
    }
    return (*handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t
hipblasDestroy(hipblasHandle_t handle)
{
    if(handle != nullptr)
    {
        {
            std::lock_guard<std::mutex> lk(g_handle_stream_mu);
            g_handle_stream.erase(handle);
        }
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
    #ifdef hipGetBackendName
    std::cerr << "Error: The hipGetBackendName API is deprecated. Please update your H4I-MKLShim to use the latest API." << std::endl;
    return HIPBLAS_STATUS_INTERNAL_ERROR;
    #endif

    //this is a context with the native handles and the NULL stream
    H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    // Obtain the backendnativehandles for the stream we want to use
    int nHandles;
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), 0, &nHandles);
    unsigned long handles[nHandles];
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), handles, 0);
    H4I::MKLShim::SetStream(ctxt, handles, nHandles);

    std::lock_guard<std::mutex> lk(g_handle_stream_mu);
    g_handle_stream[handle] = stream;
  }
  return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t
hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId)
{
    if (handle == nullptr)
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    if (streamId == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;

    std::lock_guard<std::mutex> lk(g_handle_stream_mu);
    auto it = g_handle_stream.find(handle);
    // If no explicit stream was set, report the default (null) stream — this
    // matches the behavior of the underlying MKLShim queue in that case.
    *streamId = (it != g_handle_stream.end()) ? it->second : nullptr;
    return HIPBLAS_STATUS_SUCCESS;
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
