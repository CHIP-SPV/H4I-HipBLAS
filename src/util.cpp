// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include <atomic>
#include "hip/hip_runtime.h"	// Necessary for CHIP-SPV implementation.
#include "hip/hip_interop.h"	// Necessary for CHIP-SPV implementation.
#include "hipblas.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hipblas/impl/util.h"
#include "h4i/hiputils/HandleManager.h"


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

static H4I::HIPUtils::HandleManager<hipblasHandle_t, hipblasStatus_t, HIPBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_HANDLE_IS_NULLPTR> hmgr;

hipblasStatus_t
hipblasCreate(hipblasHandle_t* handle)
{
    return hmgr.Create(handle);
}

hipblasStatus_t
hipblasDestroy(hipblasHandle_t handle)
{
    return hmgr.Destroy(handle);
}

hipblasStatus_t
hipblasSetStream(hipblasHandle_t handle, hipStream_t stream)
{
    return hmgr.SetStream(handle, stream);
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
