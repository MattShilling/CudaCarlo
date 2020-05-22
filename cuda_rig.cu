#include "cuda_rig.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

void CudaRig::Init(int num_threads) {
  if (mp_good_) {
    // Set the number of threads we want to use.
    // omp_set_num_threads(num_threads);
    // fprintf(stderr, "Using %d threads\n", num_threads);
    // We aren't using OMP multithreading for this..yet.

    // Initialize test memory.
    test_init_(mem_);

    // Clear our records.
    perf_.clear();
  } else {
    printf("Init error: OpenMP not supported!\n");
  }
}

void CudaRig::Run(double ops) {
  // Get the starting time for our test.
  double start = omp_get_wtime();
  // Run. That. Test!
  test_run_(mem_);
  // Get the ending time for our test.
  double stop = omp_get_wtime();
  // Calculate the multiplications per second we accomplished.
  double time_per_op = ops / (stop - start);
  // Convert into megamults.
  double perf = time_per_op / 1000000.00;
  // Add results to our records.
  perf_.push_back(perf);
}

double CudaRig::MaxPerf() {
  return *std::max_element(perf_.begin(), perf_.end());
}

double CudaRig::MinPerf() {
  return *std::min_element(perf_.begin(), perf_.end());
}

static int CudaRig::AddCopy(void **device, void *host, size_t sz) {
  cudaError_t status;
  status = cudaMalloc(device, sz);
  checkCudaErrors(status);
  // Copy host memory to the GPU.
  status =
      cudaMemcpy(device, host, sz, cudaMemcpyHostToDevice);
  checkCudaErrors(status);
}