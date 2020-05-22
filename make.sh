#!/bin/bash

# I tried so hard to get CMake working but :(

/usr/local/apps/cuda/cuda-10.1/bin/nvcc -o CudaCarlo cuda_carlo.cu -std=c++11 -I/usr/local/apps/cuda/cuda-10.1/samples/common/inc/
