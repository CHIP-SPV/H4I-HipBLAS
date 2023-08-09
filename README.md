<!---
Copyright 2021-2023 UT-Battelle
See LICENSE.txt in the root of the source distribution for license info.
-->

# Overview

This project provides a library that exposes the HipBLAS interface
and whose functions are implemented to run on Intel GPUs via the
SYCL version of the Intel MKL library.

It relies on a separate MKL shim library that calls the MKL
implementations. It can be found [here](https://github.com/CHIP-SPV/H4I-MKLShim).

# Building

Example:

    mkdir build && cd build
    . /opt/intel/oneapi/setvars.sh
    cmake .. -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_INSTALL_PREFIX=$HOME/local/stow/H4I-HipBLAS
    make all install


