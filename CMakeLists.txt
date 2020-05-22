 
cmake_minimum_required(VERSION 2.8)

# Set the project name.
#project(CudaCarlo CXX)
#set(TARGET CudaCarlo CMAKE_CXX_STANDARD 11)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/apps/cuda/cuda-10.1/")
find_package(CUDA QUIET REQUIRED)

# add_compile_options(-std=c++11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

# add the executable
cuda_add_executable(CudaCarlo cuda_carlo.cu helper_cuda.h helper_functions.h helper_image.h helper_string.h helper_timer.h exception.h) 
