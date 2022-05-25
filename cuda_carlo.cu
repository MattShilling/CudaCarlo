// System includes
#include <assert.h>
#include <chrono>
#include <malloc.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"

#include "cuda_rig.h"

// setting the number of trials in the monte carlo simulation:
#ifndef NUMTRIALS
#define NUMTRIALS (1024 * 1024)
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 32 // number of threads per block
#endif

#define NUMBLOCKS (NUMTRIALS / BLOCKSIZE)

// ranges for the random numbers:
const float XCMIN = 0.5;
const float XCMAX = 2.0;
const float YCMIN = 0.5;
const float YCMAX = 2.0;
const float RMIN = 0.5;
const float RMAX = 2.0;

struct Memory {
  // Allocate host memory.
  float *xcs;
  float *ycs;
  float *rs;
  int *hits;
};

struct TestMem {
  Memory *device;
  Memory *host;
  bool initialized;
  int num_trials;

  TestMem() {
    device = new Memory();
    host = new Memory();
  }

  ~TestMem() {
    // Clean up memory.
    delete[] host->xcs;
    delete[] host->ycs;
    delete[] host->rs;
    delete[] host->hits;

    cudaError_t status;
    status = cudaFree(device->xcs);
    status = cudaFree(device->ycs);
    status = cudaFree(device->rs);
    status = cudaFree(device->hits);
    checkCudaErrors(status);

    delete device;
    delete host;
  }
};

__global__ void MonteCarlo(float *xcs, float *ycs, float *rs, int *hits) {
  unsigned int wgNumber = blockIdx.x;
  unsigned int wgDimension = blockDim.x;
  unsigned int threadNum = threadIdx.x;
  unsigned int gid = wgNumber * wgDimension + threadNum;
  // all the monte carlo stuff goes in here
  // if we make it all the way through, then Hits[gid] = 1

  // randomize the location and radius of the circle:
  float xc = xcs[gid];
  float yc = ycs[gid];
  float r = rs[gid];

  float k_tn = tanf(static_cast<float>((M_PI / 180.0) * 30.0));
  hits[gid] = 0;

  // solve for the intersection using the quadratic formula:
  float a = 1.0 + k_tn * k_tn;
  float b = -2.0 * (xc + yc * k_tn);
  float c = xc * xc + yc * yc - r * r;
  float d = b * b - 4.0 * a * c;

  if (d >= 0.0) {
    // hits the circle:
    // get the first intersection:
    d = sqrt(d);
    float t1 = (-b + d) / (2. * a); // time to intersect the circle
    float t2 = (-b - d) / (2. * a); // time to intersect the circle
    float tmin = t1 < t2 ? t1 : t2; // only care about the first intersection

    if (tmin >= 0.0) {
      // where does it intersect the circle?
      float xcir = tmin;
      float ycir = tmin * k_tn;

      // get the unitized normal vector at the point of intersection:
      float nx = xcir - xc;
      float ny = ycir - yc;
      float nxy = sqrt(nx * nx + ny * ny);
      nx /= nxy; // unit vector
      ny /= nxy; // unit vector

      // get the unitized incoming vector:
      float inx = xcir - 0.0;
      float iny = ycir - 0.0;
      float in = sqrt(inx * inx + iny * iny);
      inx /= in; // unit vector
      iny /= in; // unit vector

      // get the outgoing (bounced) vector:
      float dot = inx * nx + iny * ny;
      float outy =
          iny - 2.0 * ny * dot; // angle of reflection = angle of incidence`

      // find out if it hits the infinite plate:
      float t = (0.0 - ycir) / outy;
      if (t >= 0.0) {
        hits[gid] = 1;
      }
    }
  }
}

void test_init(void *mem) {
  TestMem *data = static_cast<TestMem *>(mem);

  // Allocate host memory.
  data->host->xcs = new float[data->num_trials];
  data->host->ycs = new float[data->num_trials];
  data->host->rs = new float[data->num_trials];
  data->host->hits = new int[data->num_trials];

  // Create uniform random generator.
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  // Most Monte Carlo sampling or integration techniques assume a “random number
  // generator,” which generates uniform statistically independent values
  // - MONTE CARLO TECHNIQUES, 2009 by G. Cowan
  std::uniform_real_distribution<float> xcs_dist(XCMIN, XCMAX);
  std::uniform_real_distribution<float> ycs_dist(YCMIN, YCMAX);
  std::uniform_real_distribution<float> rs_dist(RMIN, RMAX);

  // Fill the random-value arrays.
  for (int n = 0; n < data->num_trials; n++) {
    data->host->xcs[n] = xcs_dist(generator);
    data->host->ycs[n] = ycs_dist(generator);
    data->host->rs[n] = rs_dist(generator);
    data->host->hits[n] = 0;
  }

  CudaRig::InitAndCopy(reinterpret_cast<void **>(&data->device->xcs),
                       data->host->xcs, data->num_trials * sizeof(float));
  CudaRig::InitAndCopy(reinterpret_cast<void **>(&data->device->ycs),
                       data->host->ycs, data->num_trials * sizeof(float));
  CudaRig::InitAndCopy(reinterpret_cast<void **>(&data->device->rs),
                       data->host->rs, data->num_trials * sizeof(float));
  CudaRig::InitAndCopy(reinterpret_cast<void **>(&data->device->hits),
                       data->host->hits, data->num_trials * sizeof(int));
}

// Main program.
int main(int argc, char *argv[]) {

  int dev = findCudaDevice(argc, (const char **)argv);

  TestMem *mem = new TestMem();
  mem->initialized = false;

  // Taking in some command line arguments to control the program.
  int block_size = BLOCKSIZE;
  int num_trials = NUMTRIALS;

  if (argc >= 2) {
    block_size = std::stoi(std::string(argv[1]));
  }

  if (argc >= 3) {
    num_trials = std::stoi(std::string(argv[2]));
  }

  mem->num_trials = num_trials;

  fprintf(stderr, "Block size = %d\n", block_size);
  fprintf(stderr, "Num trials size = %d\n", num_trials);

  CudaRig cuda_carlo(mem, test_init);
  cuda_carlo.Init();

  // Set up the execution parameters.
  dim3 threads(block_size, 1, 1);

  // Set the number of blocks.
  int num_blocks = (num_trials / block_size);
  dim3 grid(num_blocks, 1, 1);

  CudaTimer t;
  CudaRig::StartCudaTimer(&t);

  // Execute the kernel.
  MonteCarlo<<<grid, threads>>>(mem->device->xcs, mem->device->ycs,
                                mem->device->rs, mem->device->hits);

  CudaRig::StopCudaTimer(&t);

  cudaError_t status;

  float msecTotal = 0.0;
  status = cudaEventElapsedTime(&msecTotal, t.start, t.stop);
  checkCudaErrors(status);

  // Compute and print the performance.
  double secondsTotal = 1e-3 * static_cast<double>(msecTotal);
  double trialsPerSecond = static_cast<double>(NUMTRIALS) / secondsTotal;
  double megaTrialsPerSecond = trialsPerSecond / 1e6;
  fprintf(stderr, "MegaTrials/Second = %10.4lf\n", megaTrialsPerSecond);

  // Copy result from the device to the host.
  static_assert(sizeof(*mem->host->hits) == sizeof(*mem->device->hits));
  status = cudaMemcpy(mem->host->hits, mem->device->hits,
                      num_trials * sizeof(int), cudaMemcpyDeviceToHost);
  checkCudaErrors(status);
  cudaDeviceSynchronize();

  // Compute the probability.
  int numHits = 0;
  for (int i = 0; i < num_trials; i++) {
    numHits += mem->host->hits[i];
  }

  printf("hits = %d\n", numHits);
  printf("num_trials = %d\n", num_trials);

  double probability =
      100.0 * static_cast<double>(numHits) / static_cast<double>(num_trials);
  fprintf(stderr, "Probability = %6.3f%%\n", probability);

  return 0;
}
