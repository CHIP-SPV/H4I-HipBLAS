// Correctness tests for the mixed precision entry points:
//   hipblasHdot
//   hipblasDotEx / hipblasDotEx_v2 (including a mixed precision rejection)
//   hipblasGemmEx for R_16F -> R_16F, R_8I -> R_32I and R_64F -> R_64F
//
// Every GPU result is validated against a CPU reference computed with the
// classic host CBLAS interface of the same oneMKL that backs the GPU path.
//
// Choice of CPU reference per test:
//   * half dot / half gemm: the operands are promoted to float and the
//     reference is cblas_sdot / cblas_sgemm.  Host CBLAS has no binary16 dot,
//     and the GPU result is rounded to half on the way out anyway, so a float
//     reference with a half sized tolerance is the tightest meaningful check.
//     All inputs are exactly representable in binary16, so the only error left
//     is the rounding of the accumulator.
//   * int8 gemm: deliberately NOT cblas_gemm_s8u8s32.  That kernel multiplies
//     signed by *unsigned* 8 bit operands (s8 x u8), whereas hipblasGemmEx
//     with HIPBLAS_R_8I inputs is s8 x s8, so every negative entry of B would
//     be reinterpreted as a large positive value and the reference would be
//     wrong.  The operands here are small (|v| <= 3 with k = 24, so
//     |C| <= 216), which is exact in a double, making cblas_dgemm on promoted
//     copies an exact integer reference.
//   * double gemm: cblas_dgemm directly.
//
// Output buffers are poisoned with a sentinel before every call, so an
// implementation that silently does nothing cannot pass a test.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <mkl_cblas.h>
#include "hipblas.h"

#define CHECK_HIP(expr) do { \
    hipError_t e = (expr); \
    if (e != hipSuccess) { \
      std::cerr << #expr " failed: " << hipGetErrorString(e) \
                << " at line " << __LINE__ << "\n"; \
      return EXIT_FAILURE; \
    } \
  } while (0)

#define CHECK_BLAS(expr) do { \
    hipblasStatus_t s = (expr); \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
      std::cerr << #expr " failed: status=" << s \
                << " at line " << __LINE__ << "\n"; \
      return EXIT_FAILURE; \
    } \
  } while (0)

// Host side binary16.  clang supports _Float16 on x86-64 and its layout is the
// IEEE 754 binary16 that hipblasHalf / sycl::half use.
using half_t = _Float16;

static const float  HALF_DOT_TOLERANCE  = 2e-3f;   // one half ulp is ~4.9e-4
static const float  HALF_GEMM_TOLERANCE = 1e-2f;
static const double DOUBLE_TOLERANCE    = 1e-12;

static const float   POISON_HALF   = -111.0f;
static const int32_t POISON_INT32  = -987654;
static const double  POISON_DOUBLE = -1.0e30;

static bool closeRel(double got, double want, double tol) {
  return std::fabs(got - want) <= tol * (1.0 + std::fabs(want));
}

int main() {
  hipblasHandle_t handle;
  CHECK_BLAS(hipblasCreate(&handle));
  CHECK_BLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

  int pass = 0, fail = 0;

  // ------------------------------------------------------------------
  // Shared half vectors for the dot tests.
  // ------------------------------------------------------------------
  const int N = 1024;
  std::vector<half_t> hx(N), hy(N);
  std::vector<float>  fx(N), fy(N);
  for (int i = 0; i < N; i++) {
    // Multiples of 1/8 and 1/4 are exact in binary16.
    float a = ((i * 37) % 13 - 6) * 0.125f;
    float b = ((i * 53) % 11 - 5) * 0.25f;
    hx[i] = (half_t)a; fx[i] = a;
    hy[i] = (half_t)b; fy[i] = b;
  }
  // CPU reference for both the half and the float dot products.
  const float dotRef = cblas_sdot((MKL_INT)N, fx.data(), 1, fy.data(), 1);

  half_t *dhx, *dhy, *dhr;
  float  *dfx, *dfy, *dfr;
  CHECK_HIP(hipMalloc(&dhx, N * sizeof(half_t)));
  CHECK_HIP(hipMalloc(&dhy, N * sizeof(half_t)));
  CHECK_HIP(hipMalloc(&dhr, sizeof(half_t)));
  CHECK_HIP(hipMalloc(&dfx, N * sizeof(float)));
  CHECK_HIP(hipMalloc(&dfy, N * sizeof(float)));
  CHECK_HIP(hipMalloc(&dfr, sizeof(float)));
  CHECK_HIP(hipMemcpy(dhx, hx.data(), N * sizeof(half_t), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dhy, hy.data(), N * sizeof(half_t), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dfx, fx.data(), N * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dfy, fy.data(), N * sizeof(float), hipMemcpyHostToDevice));

  const half_t poisonHalf = (half_t)POISON_HALF;

  // --- hipblasHdot vs cblas_sdot ---
  {
    CHECK_HIP(hipMemcpy(dhr, &poisonHalf, sizeof(half_t), hipMemcpyHostToDevice));
    CHECK_BLAS(hipblasHdot(handle, N, (const hipblasHalf*)dhx, 1,
                           (const hipblasHalf*)dhy, 1, (hipblasHalf*)dhr));
    CHECK_HIP(hipDeviceSynchronize());
    half_t out;
    CHECK_HIP(hipMemcpy(&out, dhr, sizeof(half_t), hipMemcpyDeviceToHost));
    float got = (float)out;
    if (got == POISON_HALF) {
      fail++; std::cerr << "FAIL Hdot: result still poisoned, no work done\n";
    } else if (!closeRel(got, dotRef, HALF_DOT_TOLERANCE)) {
      fail++; std::cerr << "FAIL Hdot: got " << got << ", cblas_sdot " << dotRef << "\n";
    } else {
      pass++; std::cout << "PASS Hdot: " << got << " vs cblas_sdot " << dotRef << "\n";
    }
  }

  // --- hipblasDotEx (HIPBLAS_R_16F) vs cblas_sdot ---
  {
    CHECK_HIP(hipMemcpy(dhr, &poisonHalf, sizeof(half_t), hipMemcpyHostToDevice));
    CHECK_BLAS(hipblasDotEx(handle, N, dhx, HIPBLAS_R_16F, 1,
                            dhy, HIPBLAS_R_16F, 1, dhr, HIPBLAS_R_16F,
                            HIPBLAS_R_16F));
    CHECK_HIP(hipDeviceSynchronize());
    half_t out;
    CHECK_HIP(hipMemcpy(&out, dhr, sizeof(half_t), hipMemcpyDeviceToHost));
    float got = (float)out;
    if (got == POISON_HALF) {
      fail++; std::cerr << "FAIL DotEx(R_16F): result still poisoned\n";
    } else if (!closeRel(got, dotRef, HALF_DOT_TOLERANCE)) {
      fail++; std::cerr << "FAIL DotEx(R_16F): got " << got << ", cblas_sdot " << dotRef << "\n";
    } else {
      pass++; std::cout << "PASS DotEx(R_16F): " << got << " vs cblas_sdot " << dotRef << "\n";
    }
  }

  // --- hipblasDotEx_v2 (HIP_R_32F) vs cblas_sdot ---
  {
    float poison = POISON_HALF;
    CHECK_HIP(hipMemcpy(dfr, &poison, sizeof(float), hipMemcpyHostToDevice));
    CHECK_BLAS(hipblasDotEx_v2(handle, N, dfx, HIP_R_32F, 1, dfy, HIP_R_32F, 1,
                               dfr, HIP_R_32F, HIP_R_32F));
    CHECK_HIP(hipDeviceSynchronize());
    float got;
    CHECK_HIP(hipMemcpy(&got, dfr, sizeof(float), hipMemcpyDeviceToHost));
    if (got == poison) {
      fail++; std::cerr << "FAIL DotEx_v2(R_32F): result still poisoned\n";
    } else if (!closeRel(got, dotRef, 1e-5)) {
      fail++; std::cerr << "FAIL DotEx_v2(R_32F): got " << got << ", cblas_sdot " << dotRef << "\n";
    } else {
      pass++; std::cout << "PASS DotEx_v2(R_32F): " << got << " vs cblas_sdot " << dotRef << "\n";
    }
  }

  // --- mixed precision must be rejected, not silently computed ---
  {
    hipblasStatus_t s = hipblasDotEx_v2(handle, N, dhx, HIP_R_16F, 1,
                                        dfy, HIP_R_32F, 1, dfr, HIP_R_32F,
                                        HIP_R_32F);
    if (s == HIPBLAS_STATUS_NOT_SUPPORTED) {
      pass++; std::cout << "PASS DotEx_v2 mixed x=16F y=32F rejected\n";
    } else {
      fail++; std::cerr << "FAIL DotEx_v2 mixed types: expected NOT_SUPPORTED, got status=" << s << "\n";
    }
  }

  CHECK_HIP(hipFree(dhx)); CHECK_HIP(hipFree(dhy)); CHECK_HIP(hipFree(dhr));
  CHECK_HIP(hipFree(dfx)); CHECK_HIP(hipFree(dfy)); CHECK_HIP(hipFree(dfr));

  // ------------------------------------------------------------------
  // GemmEx: column major, no transpose, lda = m, ldb = k, ldc = m.
  // ------------------------------------------------------------------
  const int M = 32, NN = 24, K = 24;

  // --- hipblasGemmEx R_16F x R_16F -> R_16F vs cblas_sgemm ---
  {
    std::vector<half_t> hA(M * K), hB(K * NN), hC(M * NN);
    std::vector<float>  fA(M * K), fB(K * NN), ref(M * NN, 0.0f);
    for (int i = 0; i < M * K; i++)  { float v = (((i * 7) % 9) - 4) * 0.25f;  hA[i] = (half_t)v; fA[i] = v; }
    for (int i = 0; i < K * NN; i++) { float v = (((i * 11) % 7) - 3) * 0.5f;  hB[i] = (half_t)v; fB[i] = v; }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (MKL_INT)M,
                (MKL_INT)NN, (MKL_INT)K, 1.0f, fA.data(), (MKL_INT)M,
                fB.data(), (MKL_INT)K, 0.0f, ref.data(), (MKL_INT)M);
    for (int i = 0; i < M * NN; i++) hC[i] = (half_t)POISON_HALF;

    half_t *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, M * K * sizeof(half_t)));
    CHECK_HIP(hipMalloc(&dB, K * NN * sizeof(half_t)));
    CHECK_HIP(hipMalloc(&dC, M * NN * sizeof(half_t)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), M * K * sizeof(half_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), K * NN * sizeof(half_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, hC.data(), M * NN * sizeof(half_t), hipMemcpyHostToDevice));

    // alpha and beta are half, matching the HIPBLAS_R_16F compute type.
    const half_t alpha = (half_t)1.0f, beta = (half_t)0.0f;
    hipblasStatus_t s = hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, NN, K,
                                      &alpha, dA, HIPBLAS_R_16F, M,
                                      dB, HIPBLAS_R_16F, K, &beta,
                                      dC, HIPBLAS_R_16F, M, HIPBLAS_R_16F,
                                      HIPBLAS_GEMM_DEFAULT);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(hC.data(), dC, M * NN * sizeof(half_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));

    int poisoned = 0, bad = 0, nonfinite = 0;
    for (int i = 0; i < M * NN; i++) {
      float got = (float)hC[i];
      if (got == POISON_HALF) { poisoned++; continue; }
      if (!std::isfinite(got)) { nonfinite++; continue; }
      if (!closeRel(got, ref[i], HALF_GEMM_TOLERANCE)) {
        if (bad == 0)
          std::cerr << "  first mismatch at " << i << ": got " << got
                    << ", cblas_sgemm " << ref[i] << "\n";
        bad++;
      }
    }
    if (s != HIPBLAS_STATUS_SUCCESS) {
      fail++; std::cerr << "FAIL GemmEx R_16F: status=" << s << "\n";
    } else if (poisoned || nonfinite || bad) {
      fail++; std::cerr << "FAIL GemmEx R_16F: poisoned=" << poisoned
                        << " non-finite=" << nonfinite << " mismatched=" << bad
                        << " of " << (M * NN) << "\n";
    } else {
      pass++; std::cout << "PASS GemmEx R_16F->R_16F: " << (M * NN)
                        << " elements match cblas_sgemm\n";
    }
  }

  // --- hipblasGemmEx R_8I x R_8I -> R_32I vs cblas_dgemm (exact) ---
  {
    std::vector<int8_t>  hA(M * K), hB(K * NN);
    std::vector<int32_t> hC(M * NN);
    std::vector<double>  dA_(M * K), dB_(K * NN), ref(M * NN, 0.0);
    for (int i = 0; i < M * K; i++)  { hA[i] = (int8_t)(((i * 7) % 7) - 3); dA_[i] = hA[i]; }
    for (int i = 0; i < K * NN; i++) { hB[i] = (int8_t)(((i * 5) % 7) - 3); dB_[i] = hB[i]; }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (MKL_INT)M,
                (MKL_INT)NN, (MKL_INT)K, 1.0, dA_.data(), (MKL_INT)M,
                dB_.data(), (MKL_INT)K, 0.0, ref.data(), (MKL_INT)M);
    for (int i = 0; i < M * NN; i++) hC[i] = POISON_INT32;

    int8_t *dA, *dB; int32_t *dC;
    CHECK_HIP(hipMalloc(&dA, M * K));
    CHECK_HIP(hipMalloc(&dB, K * NN));
    CHECK_HIP(hipMalloc(&dC, M * NN * sizeof(int32_t)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), M * K, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), K * NN, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, hC.data(), M * NN * sizeof(int32_t), hipMemcpyHostToDevice));

    // alpha and beta are int32_t, matching the HIPBLAS_R_32I compute type.
    const int32_t alpha = 1, beta = 0;
    hipblasStatus_t s = hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, NN, K,
                                      &alpha, dA, HIPBLAS_R_8I, M,
                                      dB, HIPBLAS_R_8I, K, &beta,
                                      dC, HIPBLAS_R_32I, M, HIPBLAS_R_32I,
                                      HIPBLAS_GEMM_DEFAULT);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(hC.data(), dC, M * NN * sizeof(int32_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));

    int poisoned = 0, bad = 0;
    for (int i = 0; i < M * NN; i++) {
      if (hC[i] == POISON_INT32) { poisoned++; continue; }
      int32_t want = (int32_t)llround(ref[i]);
      if (hC[i] != want) {
        if (bad == 0)
          std::cerr << "  first mismatch at " << i << ": got " << hC[i]
                    << ", cblas_dgemm " << want << "\n";
        bad++;
      }
    }
    if (s != HIPBLAS_STATUS_SUCCESS) {
      fail++; std::cerr << "FAIL GemmEx R_8I->R_32I: status=" << s << "\n";
    } else if (poisoned || bad) {
      fail++; std::cerr << "FAIL GemmEx R_8I->R_32I: poisoned=" << poisoned
                        << " mismatched=" << bad << " of " << (M * NN) << "\n";
    } else {
      pass++; std::cout << "PASS GemmEx R_8I->R_32I: " << (M * NN)
                        << " elements match cblas_dgemm exactly\n";
    }
  }

  // --- hipblasGemmEx R_64F x R_64F -> R_64F vs cblas_dgemm ---
  {
    std::vector<double> hA(M * K), hB(K * NN), hC(M * NN), ref(M * NN, 0.0);
    for (int i = 0; i < M * K; i++)  hA[i] = ((i * 7) % 9 - 4) * 0.25;
    for (int i = 0; i < K * NN; i++) hB[i] = ((i * 11) % 7 - 3) * 0.5;
    // alpha = 2.0 exercises the double scale factor, which cannot survive
    // being read as a float.
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (MKL_INT)M,
                (MKL_INT)NN, (MKL_INT)K, 2.0, hA.data(), (MKL_INT)M,
                hB.data(), (MKL_INT)K, 0.0, ref.data(), (MKL_INT)M);
    for (int i = 0; i < M * NN; i++) hC[i] = POISON_DOUBLE;

    double *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, M * K * sizeof(double)));
    CHECK_HIP(hipMalloc(&dB, K * NN * sizeof(double)));
    CHECK_HIP(hipMalloc(&dC, M * NN * sizeof(double)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), M * K * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), K * NN * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, hC.data(), M * NN * sizeof(double), hipMemcpyHostToDevice));

    const double alpha = 2.0, beta = 0.0;
    hipblasStatus_t s = hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, NN, K,
                                      &alpha, dA, HIPBLAS_R_64F, M,
                                      dB, HIPBLAS_R_64F, K, &beta,
                                      dC, HIPBLAS_R_64F, M, HIPBLAS_R_64F,
                                      HIPBLAS_GEMM_DEFAULT);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(hC.data(), dC, M * NN * sizeof(double), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));

    int poisoned = 0, bad = 0;
    for (int i = 0; i < M * NN; i++) {
      if (hC[i] == POISON_DOUBLE) { poisoned++; continue; }
      if (!closeRel(hC[i], ref[i], DOUBLE_TOLERANCE)) {
        if (bad == 0)
          std::cerr << "  first mismatch at " << i << ": got " << hC[i]
                    << ", cblas_dgemm " << ref[i] << "\n";
        bad++;
      }
    }
    if (s != HIPBLAS_STATUS_SUCCESS) {
      fail++; std::cerr << "FAIL GemmEx R_64F: status=" << s << "\n";
    } else if (poisoned || bad) {
      fail++; std::cerr << "FAIL GemmEx R_64F: poisoned=" << poisoned
                        << " mismatched=" << bad << " of " << (M * NN) << "\n";
    } else {
      pass++; std::cout << "PASS GemmEx R_64F->R_64F: " << (M * NN)
                        << " elements match cblas_dgemm\n";
    }
  }

  CHECK_BLAS(hipblasDestroy(handle));
  std::cout << "\n" << pass << " passed, " << fail << " failed out of "
            << (pass + fail) << " tests\n";
  return fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
