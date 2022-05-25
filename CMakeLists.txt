cmake_minimum_required(VERSION 2.8)

project(CudaCarlo CXX CUDA)

# This is the CUDA Toolkit version installed locally.
find_package(CUDA 11.7 REQUIRED)

# Target RTX Ampere cards. 
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_87")

# Compile to the C++17 standard.
list(APPEND CUDA_NVCC_FLAGS "-std=c++17")

cuda_add_executable(CudaCarlo.bin
    cuda_carlo.cu 
    cuda_rig.cu) 
