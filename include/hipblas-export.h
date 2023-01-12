// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#pragma once

// We use the ROCm hipblas.h header, unchanged.
// That header expects the HIPBLAS_EXPORT macro
// to be defined, and includes the hipblas-export.h
// header to do it.
#define HIPBLAS_EXPORT

