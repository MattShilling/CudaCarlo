#ifndef CUDA_RIG_H_
#define CUDA_RIG_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

class CudaRig {
public:
  CudaRig(std::shared_ptr<void> mem,
          std::function<void(std::shared_ptr<void> mem)> run,
          std::function<void(std::shared_ptr<void> mem)> init)
      : mem_(mem), test_init_(init), test_run_(run)){};
  void Init(int num_threads);
  void Run(double sz);

  double MaxPerf();
  double MinPerf();

  static int AddCopy();

private:
  std::shared_ptr<void> mem_;
  std::function<void(std::shared_ptr<void> mem)> test_init_;
  std::function<void(std::shared_ptr<void> mem)> test_run_;
  std::vector<double> perf_;
};

#endif // CUDA_RIG_H_