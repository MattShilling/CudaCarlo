 
cmake_minimum_required(VERSION 2.8)

# Info on making his CMakeLists.txt found on:
# https://devblogs.nvidia.com/building-cuda-applications-cmake/

# Set the project name.
project(CudaCarlo LANGUAGES CXX CUDA)

#add_compile_options(-std=c++11)

target_compile_features(CudaCarlo PUBLIC cxx_std_11)

# add the executable
add_executable(CudaCarlo cuda_carlo.cu)

set_target_properties( CudaCarlo
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# target_link_libraries(VectorizeSSE PRIVATE "${OpenMP_CXX_FLAGS}")
# target_compile_options(VectorizeSSE PRIVATE "${OpenMP_CXX_FLAGS}")