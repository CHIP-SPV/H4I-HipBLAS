// Test ILP64 (_64) Level 1 BLAS functions.
// Runs each _64 function on small vectors and verifies results against
// known-good values computed on the host.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <hip/hip_runtime.h>
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

static bool near(double a, double b, double tol = 1e-12) {
  return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

int main() {
  const int64_t N = 8;
  // Host vectors
  double hx[N], hy[N];
  for (int64_t i = 0; i < N; i++) {
    hx[i] = (double)(i + 1);          // 1,2,3,...,8
    hy[i] = (double)(N - i);           // 8,7,6,...,1
  }

  // Device vectors
  double *dx, *dy;
  CHECK_HIP(hipMalloc(&dx, N * sizeof(double)));
  CHECK_HIP(hipMalloc(&dy, N * sizeof(double)));
  CHECK_HIP(hipMemcpy(dx, hx, N * sizeof(double), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dy, hy, N * sizeof(double), hipMemcpyHostToDevice));

  hipblasHandle_t handle;
  CHECK_BLAS(hipblasCreate(&handle));
  CHECK_BLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

  int pass = 0, fail = 0;

  // Probe fp64 support: Intel Arc (Alchemist) and similar devices lack native fp64.
  // MKL returns HIPBLAS_STATUS_EXECUTION_FAILED for double ops on unsupported devices.
  bool fp64_supported = true;

  // --- Dasum_64 ---
  {
    double result = 0;
    hipblasStatus_t s = hipblasDasum_64(handle, N, dx, 1, &result);
    if (s == HIPBLAS_STATUS_EXECUTION_FAILED) {
      std::cout << "NOTE: device does not support fp64 — skipping double precision tests\n";
      fp64_supported = false;
    } else if (s != HIPBLAS_STATUS_SUCCESS) {
      std::cerr << "hipblasDasum_64 failed: status=" << s << " at line " << __LINE__ << "\n";
      return EXIT_FAILURE;
    } else {
      // sum |1|+|2|+...+|8| = 36
      if (near(result, 36.0)) { pass++; std::cout << "PASS Dasum_64: " << result << "\n"; }
      else { fail++; std::cerr << "FAIL Dasum_64: expected 36, got " << result << "\n"; }
    }
  }

  if (fp64_supported) {
    // --- Dnrm2_64 ---
    {
      double result = 0;
      CHECK_BLAS(hipblasDnrm2_64(handle, N, dx, 1, &result));
      // sqrt(1+4+9+16+25+36+49+64) = sqrt(204)
      double expected = std::sqrt(204.0);
      if (near(result, expected)) { pass++; std::cout << "PASS Dnrm2_64: " << result << "\n"; }
      else { fail++; std::cerr << "FAIL Dnrm2_64: expected " << expected << ", got " << result << "\n"; }
    }

    // --- Idamax_64 ---
    {
      int64_t result = -1;
      CHECK_BLAS(hipblasIdamax_64(handle, N, dx, 1, &result));
      // max element is 8 at index 7 (0-based) -> hipBLAS returns 1-based = 8
      if (result == 8) { pass++; std::cout << "PASS Idamax_64: " << result << "\n"; }
      else { fail++; std::cerr << "FAIL Idamax_64: expected 8, got " << result << "\n"; }
    }

    // --- Dscal_64 ---
    {
      // Scale dy by 2.0: [8,7,...,1] -> [16,14,...,2]
      CHECK_HIP(hipMemcpy(dy, hy, N * sizeof(double), hipMemcpyHostToDevice));
      double alpha = 2.0;
      CHECK_BLAS(hipblasDscal_64(handle, N, &alpha, dy, 1));
      double hout[N];
      CHECK_HIP(hipMemcpy(hout, dy, N * sizeof(double), hipMemcpyDeviceToHost));
      bool ok = true;
      for (int64_t i = 0; i < N; i++) {
        if (!near(hout[i], 2.0 * hy[i])) { ok = false; break; }
      }
      if (ok) { pass++; std::cout << "PASS Dscal_64: [" << hout[0] << "," << hout[1] << ",...," << hout[N-1] << "]\n"; }
      else { fail++; std::cerr << "FAIL Dscal_64\n"; }
    }

    // --- Dcopy_64 ---
    {
      double *dz;
      CHECK_HIP(hipMalloc(&dz, N * sizeof(double)));
      CHECK_HIP(hipMemset(dz, 0, N * sizeof(double)));
      CHECK_BLAS(hipblasDcopy_64(handle, N, dx, 1, dz, 1));
      double hout[N];
      CHECK_HIP(hipMemcpy(hout, dz, N * sizeof(double), hipMemcpyDeviceToHost));
      bool ok = true;
      for (int64_t i = 0; i < N; i++) {
        if (!near(hout[i], hx[i])) { ok = false; break; }
      }
      if (ok) { pass++; std::cout << "PASS Dcopy_64\n"; }
      else { fail++; std::cerr << "FAIL Dcopy_64\n"; }
      CHECK_HIP(hipFree(dz));
    }

    // --- Daxpy_64 ---
    {
      // y = alpha*x + y, alpha=3.0, x=[1..8], y=[8..1] -> [11,13,15,17,19,21,23,25]
      CHECK_HIP(hipMemcpy(dy, hy, N * sizeof(double), hipMemcpyHostToDevice));
      double alpha = 3.0;
      CHECK_BLAS(hipblasDaxpy_64(handle, N, &alpha, dx, 1, dy, 1));
      double hout[N];
      CHECK_HIP(hipMemcpy(hout, dy, N * sizeof(double), hipMemcpyDeviceToHost));
      bool ok = true;
      for (int64_t i = 0; i < N; i++) {
        double expected = 3.0 * hx[i] + hy[i];
        if (!near(hout[i], expected)) { ok = false; std::cerr << "  i=" << i << " expected=" << expected << " got=" << hout[i] << "\n"; break; }
      }
      if (ok) { pass++; std::cout << "PASS Daxpy_64: [" << hout[0] << "," << hout[1] << ",...," << hout[N-1] << "]\n"; }
      else { fail++; std::cerr << "FAIL Daxpy_64\n"; }
    }
  }

  // --- Now float variants ---

  float hxf[N], hyf[N];
  for (int64_t i = 0; i < N; i++) {
    hxf[i] = (float)(i + 1);
    hyf[i] = (float)(N - i);
  }
  float *dxf, *dyf;
  CHECK_HIP(hipMalloc(&dxf, N * sizeof(float)));
  CHECK_HIP(hipMalloc(&dyf, N * sizeof(float)));
  CHECK_HIP(hipMemcpy(dxf, hxf, N * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dyf, hyf, N * sizeof(float), hipMemcpyHostToDevice));

  // --- Sasum_64 ---
  {
    float result = 0;
    CHECK_BLAS(hipblasSasum_64(handle, N, dxf, 1, &result));
    if (near(result, 36.0f, 1e-5)) { pass++; std::cout << "PASS Sasum_64: " << result << "\n"; }
    else { fail++; std::cerr << "FAIL Sasum_64: expected 36, got " << result << "\n"; }
  }

  // --- Snrm2_64 ---
  {
    float result = 0;
    CHECK_BLAS(hipblasSnrm2_64(handle, N, dxf, 1, &result));
    float expected = std::sqrt(204.0f);
    if (near(result, expected, 1e-5)) { pass++; std::cout << "PASS Snrm2_64: " << result << "\n"; }
    else { fail++; std::cerr << "FAIL Snrm2_64: expected " << expected << ", got " << result << "\n"; }
  }

  // --- Isamax_64 ---
  {
    int64_t result = -1;
    CHECK_BLAS(hipblasIsamax_64(handle, N, dxf, 1, &result));
    if (result == 8) { pass++; std::cout << "PASS Isamax_64: " << result << "\n"; }
    else { fail++; std::cerr << "FAIL Isamax_64: expected 8, got " << result << "\n"; }
  }

  // --- Sscal_64 ---
  {
    CHECK_HIP(hipMemcpy(dyf, hyf, N * sizeof(float), hipMemcpyHostToDevice));
    float alpha = 2.0f;
    CHECK_BLAS(hipblasSscal_64(handle, N, &alpha, dyf, 1));
    float hout[N];
    CHECK_HIP(hipMemcpy(hout, dyf, N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t i = 0; i < N; i++) {
      if (!near(hout[i], 2.0f * hyf[i], 1e-5)) { ok = false; break; }
    }
    if (ok) { pass++; std::cout << "PASS Sscal_64\n"; }
    else { fail++; std::cerr << "FAIL Sscal_64\n"; }
  }

  // --- Scopy_64 ---
  {
    float *dzf;
    CHECK_HIP(hipMalloc(&dzf, N * sizeof(float)));
    CHECK_HIP(hipMemset(dzf, 0, N * sizeof(float)));
    CHECK_BLAS(hipblasScopy_64(handle, N, dxf, 1, dzf, 1));
    float hout[N];
    CHECK_HIP(hipMemcpy(hout, dzf, N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t i = 0; i < N; i++) {
      if (!near(hout[i], hxf[i], 1e-5)) { ok = false; break; }
    }
    if (ok) { pass++; std::cout << "PASS Scopy_64\n"; }
    else { fail++; std::cerr << "FAIL Scopy_64\n"; }
    CHECK_HIP(hipFree(dzf));
  }

  // --- Saxpy_64 ---
  {
    CHECK_HIP(hipMemcpy(dyf, hyf, N * sizeof(float), hipMemcpyHostToDevice));
    float alpha = 3.0f;
    CHECK_BLAS(hipblasSaxpy_64(handle, N, &alpha, dxf, 1, dyf, 1));
    float hout[N];
    CHECK_HIP(hipMemcpy(hout, dyf, N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t i = 0; i < N; i++) {
      float expected = 3.0f * hxf[i] + hyf[i];
      if (!near(hout[i], expected, 1e-5)) { ok = false; break; }
    }
    if (ok) { pass++; std::cout << "PASS Saxpy_64\n"; }
    else { fail++; std::cerr << "FAIL Saxpy_64\n"; }
  }

  // --- Snrm2_64 ---
  {
    float result = 0;
    CHECK_BLAS(hipblasSnrm2_64(handle, N, dxf, 1, &result));
    float expected = std::sqrt(204.0f);
    if (near(result, expected, 1e-5)) { pass++; std::cout << "PASS Snrm2_64: " << result << "\n"; }
    else { fail++; std::cerr << "FAIL Snrm2_64: expected " << expected << ", got " << result << "\n"; }
  }

  // --- Isamax_64 ---
  {
    int64_t result = -1;
    CHECK_BLAS(hipblasIsamax_64(handle, N, dxf, 1, &result));
    if (result == 8) { pass++; std::cout << "PASS Isamax_64: " << result << "\n"; }
    else { fail++; std::cerr << "FAIL Isamax_64: expected 8, got " << result << "\n"; }
  }

  // --- Sscal_64 ---
  {
    CHECK_HIP(hipMemcpy(dyf, hyf, N * sizeof(float), hipMemcpyHostToDevice));
    float alpha = 2.0f;
    CHECK_BLAS(hipblasSscal_64(handle, N, &alpha, dyf, 1));
    float hout[N];
    CHECK_HIP(hipMemcpy(hout, dyf, N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t i = 0; i < N; i++) {
      if (!near(hout[i], 2.0f * hyf[i], 1e-5)) { ok = false; break; }
    }
    if (ok) { pass++; std::cout << "PASS Sscal_64\n"; }
    else { fail++; std::cerr << "FAIL Sscal_64\n"; }
  }

  // Cleanup
  CHECK_HIP(hipFree(dxf));
  CHECK_HIP(hipFree(dyf));
  CHECK_HIP(hipFree(dx));
  CHECK_HIP(hipFree(dy));
  CHECK_BLAS(hipblasDestroy(handle));

  std::cout << "\n" << pass << " passed, " << fail << " failed out of " << (pass + fail) << " tests\n";
  return fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
