// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

#include <unordered_map>

inline
H4I::MKLShim::Operation
ToMKLShimOp(hipblasOperation_t op)
{
    std::unordered_map<hipblasOperation_t, H4I::MKLShim::Operation> map =
    {
        {HIPBLAS_OP_N, H4I::MKLShim::N},
        {HIPBLAS_OP_T, H4I::MKLShim::T},
        {HIPBLAS_OP_C, H4I::MKLShim::C}
    };

    return map[op];
}

