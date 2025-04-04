cmake_minimum_required(VERSION 3.10)
project(LD_GPU_Matching CUDA CXX)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${MPI_INCLUDE_PATH})
include_directories(/usr/local/include)  # NCCL include path

# Add link directories
link_directories(/usr/local/lib)  # NCCL lib path

# Add source files
set(SOURCES
    src/graph.cu
    src/ld_gpu.cu
    src/main.cu
)

# Create executable
add_executable(ld_gpu_matching ${SOURCES})
set_target_properties(ld_gpu_matching PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(ld_gpu_matching ${CUDA_LIBRARIES} ${MPI_LIBRARIES} nccl OpenMP::OpenMP_CXX)
