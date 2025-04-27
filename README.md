# LD-GPU: Locally Dominant Pointer-Based GPU Implementation for Maximum Weighted Matching

This is an implementation of the LD-GPU algorithm described in the paper "A Work-Efficient GPU Algorithm for Maximum Cardinality and Weighted Matching".

## Building the Project

### Prerequisites

- CUDA Toolkit (tested with CUDA 12.4)
- CMake (version 3.10 or higher)
- Boost libraries
- NCCL (NVIDIA Collective Communications Library)

### Build Instructions

```bash
mkdir -p build
cd build
cmake ..
make -j4
