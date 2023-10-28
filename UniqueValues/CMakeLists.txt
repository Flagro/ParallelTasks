cmake_minimum_required(VERSION 3.10)

# Set the project name
project(UniqueFinder CUDA CXX)

# Ensure -std=c++14 for host code and enable support for CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60")

# Add the executable and its dependencies
add_executable(UniqueFinder main.cpp unique_finder.cu)
