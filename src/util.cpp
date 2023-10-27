// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include <atomic>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hip/hip_interop.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hipblas/impl/util.h"

bool is_mkl_verion_gt_2023_0_2 = false; // Indicates current mkl version >= 2013.0.2
void getMKLVersion() {
    H4I::MKLShim::MKL_VERSION mkl_version = H4I::MKLShim::get_mkl_version();
    if (mkl_version.major > 2023){
        is_mkl_verion_gt_2023_0_2 = true;
        return;
    } else if(mkl_version.major == 2023) {
        if (mkl_version.minor > 0) {
            is_mkl_verion_gt_2023_0_2 = true;
            return;
        } else if (mkl_version.minor == 0){
            if (mkl_version.patch >= 2){
                is_mkl_verion_gt_2023_0_2 = true;
                return;
            }
        }
    }
    is_mkl_verion_gt_2023_0_2 = false;
}

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

hipblasStatus_t
hipblasCreate(hipblasHandle_t* handle)
{
    // Update mkl version first as it will be needed to decide which WA to use
    getMKLVersion();

    if(handle != nullptr)
    {
        // HIP supports mutile backends hence query current backend name
        auto backendName = hipGetBackendName();
        // Obtain the handles to the back handlers.
        unsigned long handles[4];
        int           nHandles = 4;
        hipGetBackendNativeHandles((uintptr_t)NULL, handles, &nHandles);
        *handle = H4I::MKLShim::Create(handles, nHandles, backendName);
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
