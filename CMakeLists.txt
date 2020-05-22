 
cmake_minimum_required(VERSION 2.8)

set(CMAKE_INSTALL_RPATH ${LIBNVVM_HOME})

if(CMAKE_SIZEOF_VOID_P STREQUAL "8")
  add_definitions(-DBUILD_64_BIT)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR YES)

# Set the project name.
project(CudaCarlo CXX)
set(TARGET CudaCarlo CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

# add_compile_options(-std=c++11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

# add the executable
CUDA_ADD_EXECUTABLE(CudaCarlo cuda_carlo.cu)

target_link_libraries(CudaCarlo ${NVVM_LIB} ${CUDA_LIB})
set_target_properties(CudaCarlo PROPERTIES LINK_FLAGS "-Wl,-rpath,${LIBNVVM_RPATH}")