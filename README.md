<!---
Copyright 2021-2023 UT-Battelle
See LICENSE.txt in the root of the source distribution for license info.
-->

# Overview

This project provides a library that exposes the HipBLAS interface
and whose functions are implemented to run on Intel GPUs via the
SYCL version of the Intel MKL library.

It relies on a separate shim library that actually depends on the MKL
implementations.

## Build Configuration

### MKL Threading Mode

You can configure the MKL threading mode using the `MKL_THREADING` CMake option:

```bash
cmake -DMKL_THREADING=intel_thread ..
```

Available options based on [Intel oneAPI MKL threading libraries](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-1/linking-with-threading-libraries.html):

- `sequential` (default): Single-threaded execution (libmkl_sequential)
  
- `intel_thread`: Intel OpenMP threading (libmkl_intel_thread) 
  - Requires libiomp5.so runtime library
  
- `gnu_thread`: GNU OpenMP threading (libmkl_gnu_thread)
  - Requires GNU OpenMP runtime library
  
- `tbb_thread`: Intel TBB threading (libmkl_tbb_thread)
  - Requires libtbb.so runtime library
  
- `pgi_thread`: PGI OpenMP threading (libmkl_pgi_thread)
  - Requires PGI OpenMP runtime library
