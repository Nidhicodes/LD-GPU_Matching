#include "../include/ld_gpu.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <cassert>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// NCCL error checking macro
#define NCCL_CHECK(call) \
    do { \
        ncclResult_t error = call; \
        if (error != ncclSuccess) { \
            std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << ncclGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Warp-level shuffle reduction
__device__ size_t warpReduceMax(size_t val, float weight, float* max_weight) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_weight = __shfl_down_sync(0xffffffff, *max_weight, offset);
        size_t other_val = __shfl_down_sync(0xffffffff, val, offset);
        
        if (other_weight > *max_weight) {
            *max_weight = other_weight;
            val = other_val;
        }
    }
    return val;
}

// Kernel for the pointing phase
__global__ void setPointersKernel(size_t* vertex_batch, size_t num_vertices_batch, 
                                 size_t* offsets, size_t* edges, float* weights,
                                 size_t* pointers, size_t* mate) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices_batch) return;
    
    size_t u = vertex_batch[idx];
    
    // Skip if already matched
    if (mate[u] != SIZE_MAX) return;
    
    size_t best_v = SIZE_MAX;
    float best_weight = -1.0f;
    
    // Process neighbors
    size_t start = offsets[u];
    size_t end = offsets[u + 1];
    
    for (size_t e = start; e < end; ++e) {
        size_t v = edges[e];
        
        // Skip if already matched
        if (mate[v] != SIZE_MAX) continue;
        
        float w = weights[e];
        if (w > best_weight) {
            best_weight = w;
            best_v = v;
        }
    }
    
    // Set pointer to heaviest available neighbor
    pointers[u] = best_v;
}

// Kernel for the matching phase
__global__ void setMatesKernel(size_t num_vertices, size_t* pointers, size_t* mate) {
    size_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    
    // Skip if already matched
    if (mate[u] != SIZE_MAX) return;
    
    size_t v = pointers[u];
    
    // Check for mutual pointing
    if (v != SIZE_MAX && pointers[v] == u) {
        mate[u] = v;
    }
}

// LD_GPU_Matcher implementation
LD_GPU_Matcher::LD_GPU_Matcher(Graph& graph, int num_gpus, int max_batches_per_device)
    : num_gpus(num_gpus), threads_per_block(256) {
    
    // Initialize host arrays
    h_pointers.resize(graph.num_vertices, SIZE_MAX);
    h_mate.resize(graph.num_vertices, SIZE_MAX);
    
    // Set up devices
    setupDevices();
    
    // Partition graph
    graph.partitionGraph(num_gpus, graph_partitions);
    
    // Create batches
    createBatches(max_batches_per_device);
    
    // Set up NCCL for multi-GPU communication
    if (num_gpus > 1) {
        setupNCCL();
    }
    
    // Allocate memory on each device
    d_pointers.resize(num_gpus, nullptr);
    d_mate.resize(num_gpus, nullptr);
    
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        // Copy graph partition to device
        graph_partitions[gpu].copyToDevice();
        
        // Allocate pointers and mate arrays on device
        CUDA_CHECK(cudaMalloc(&d_pointers[gpu], sizeof(size_t) * graph.num_vertices));
        CUDA_CHECK(cudaMalloc(&d_mate[gpu], sizeof(size_t) * graph.num_vertices));
        
        // Initialize arrays
        CUDA_CHECK(cudaMemset(d_pointers[gpu], 0xFF, sizeof(size_t) * graph.num_vertices)); // Set to SIZE_MAX
        CUDA_CHECK(cudaMemset(d_mate[gpu], 0xFF, sizeof(size_t) * graph.num_vertices));     // Set to SIZE_MAX
        
        // Create streams
        streams[0].resize(num_gpus);
        streams[1].resize(num_gpus);
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaStreamCreate(&streams[s][gpu]));
        }
    }
}

LD_GPU_Matcher::~LD_GPU_Matcher() {
    // Clean up device memory
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        if (d_pointers[gpu]) {
            CUDA_CHECK(cudaFree(d_pointers[gpu]));
        }
        
        if (d_mate[gpu]) {
            CUDA_CHECK(cudaFree(d_mate[gpu]));
        }
        
        // Free graph partition memory
        graph_partitions[gpu].freeDeviceMemory();
        
        // Destroy streams
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaStreamDestroy(streams[s][gpu]));
        }
    }
    
    // Clean up NCCL
    if (num_gpus > 1) {
        cleanupNCCL();
    }
}

void LD_GPU_Matcher::setupDevices() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (num_gpus > device_count) {
        std::cerr << "Requested " << num_gpus << " GPUs, but only " << device_count << " are available." << std::endl;
        num_gpus = device_count;
    }
    
    std::cout << "Using " << num_gpus << " GPUs for matching." << std::endl;
}

void LD_GPU_Matcher::createBatches(int max_batches_per_device) {
    batch_offsets.resize(num_gpus);
    
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        const Graph& partition = graph_partitions[gpu];
        size_t vertices_per_batch = (partition.num_vertices + max_batches_per_device - 1) / max_batches_per_device;
        
        std::vector<size_t>& gpu_batches = batch_offsets[gpu];
        gpu_batches.push_back(0);
        
        for (size_t v = vertices_per_batch; v < partition.num_vertices; v += vertices_per_batch) {
            gpu_batches.push_back(v);
        }
        
        gpu_batches.push_back(partition.num_vertices);
        
        std::cout << "GPU " << gpu << " has " << (gpu_batches.size() - 1) << " batches." << std::endl;
    }
}

void LD_GPU_Matcher::setupNCCL() {
    comms.resize(num_gpus);
    int devices[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        devices[i] = i;
    }
    
    // Initialize NCCL communicators
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devices));
}

void LD_GPU_Matcher::cleanupNCCL() {
    for (int i = 0; i < num_gpus; ++i) {
        ncclCommDestroy(comms[i]);
    }
}

bool LD_GPU_Matcher::executeIterationBatched() {
    bool new_matches = false;
    
    // Reset pointers
    for (size_t i = 0; i < h_pointers.size(); ++i) {
        h_pointers[i] = SIZE_MAX;
    }
    
    // Pointing phase
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        // Copy the current matching to the device
        CUDA_CHECK(cudaMemcpy(d_mate[gpu], h_mate.data(), sizeof(size_t) * h_mate.size(), cudaMemcpyHostToDevice));
        
        const Graph& partition = graph_partitions[gpu];
        const std::vector<size_t>& gpu_batches = batch_offsets[gpu];
        
        // Process each batch
        for (size_t b = 0; b < gpu_batches.size() - 1; ++b) {
            int stream_idx = b % 2;
            cudaStream_t& stream = streams[stream_idx][gpu];
            
            size_t batch_start = gpu_batches[b];
            size_t batch_end = gpu_batches[b + 1];
            size_t batch_size = batch_end - batch_start;
            
            // Create batch of vertices
            std::vector<size_t> vertex_batch(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                vertex_batch[i] = batch_start + i;
            }
            
            // Allocate memory for batch on device
            size_t* d_vertex_batch;
            CUDA_CHECK(cudaMalloc(&d_vertex_batch, sizeof(size_t) * batch_size));
            CUDA_CHECK(cudaMemcpyAsync(d_vertex_batch, vertex_batch.data(), sizeof(size_t) * batch_size, 
                                       cudaMemcpyHostToDevice, stream));
            
            // Launch kernel for pointing phase
            int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
            setPointersKernel<<<blocks, threads_per_block, 0, stream>>>(
                d_vertex_batch, batch_size,
                partition.d_offsets, partition.d_edges, partition.d_weights,
                d_pointers[gpu], d_mate[gpu]
            );
            
            // Free batch memory
            CUDA_CHECK(cudaFree(d_vertex_batch));
            
            // Synchronize stream
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        
        // Copy pointers back to host
        CUDA_CHECK(cudaMemcpy(h_pointers.data(), d_pointers[gpu], sizeof(size_t) * h_pointers.size(), cudaMemcpyDeviceToHost));
    }
    
    // Matching phase
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        const Graph& partition = graph_partitions[gpu];
        
        // Launch kernel for matching phase
        int blocks = (partition.num_vertices + threads_per_block - 1) / threads_per_block;
        setMatesKernel<<<blocks, threads_per_block>>>(
            partition.num_vertices,
            d_pointers[gpu], d_mate[gpu]
        );
        
        // Synchronize
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy updated matching back to host
        CUDA_CHECK(cudaMemcpy(h_mate.data(), d_mate[gpu], sizeof(size_t) * h_mate.size(), cudaMemcpyDeviceToHost));
    }
    
    // Check if any new matches were found
    for (size_t i = 0; i < h_mate.size(); ++i) {
        if (h_mate[i] != SIZE_MAX && h_pointers[i] != SIZE_MAX) {
            new_matches = true;
            break;
        }
    }
    
    return new_matches;
}

void LD_GPU_Matcher::computeMatching() {
    int iteration = 0;
    bool continue_matching = true;
    
    while (continue_matching) {
        std::cout << "Starting iteration " << iteration << std::endl;
        
        continue_matching = executeIterationBatched();
        
        if (!continue_matching) {
            std::cout << "No new matches found in iteration " << iteration << std::endl;
            break;
        }
        
        iteration++;
    }
    
    // Count matches
    size_t num_matches = 0;
    for (size_t i = 0; i < h_mate.size(); ++i) {
        if (h_mate[i] != SIZE_MAX) {
            num_matches++;
        }
    }
    
    std::cout << "Matching completed in " << iteration << " iterations." << std::endl;
    std::cout << "Found " << num_matches / 2 << " matched pairs." << std::endl;
}

const std::vector<size_t>& LD_GPU_Matcher::getMatching() const {
    return h_mate;
}
