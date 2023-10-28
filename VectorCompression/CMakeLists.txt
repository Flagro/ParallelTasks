cmake_minimum_required(VERSION 3.10)

# Name of your project
project(CompressedDataProject)

# Ensure C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

    # Check if the compiler is GCC or Clang to set the OpenMP version flag
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")
    endif()
endif()

# Add your source files
add_executable(CompressedDataApp
    main.cpp
    compressed_data.cpp
    huffman_data.cpp
)

# Link the OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(CompressedDataApp PUBLIC OpenMP::OpenMP_CXX)
endif()
