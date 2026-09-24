cmake_minimum_required(VERSION 3.18)

project(my_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)
find_package(CUDAToolkit REQUIRED)
add_executable(runner runner.cu)
target_link_libraries(runner PRIVATE CUDA::cublas)
