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

