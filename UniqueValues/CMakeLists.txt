cmake_minimum_required(VERSION 3.10)
project(UniqueFinder CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 60)
add_executable(UniqueFinder main.cu unique_finder.cu)
