#include "../include/ld_gpu.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <cassert>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t error = call;                                                 \
        if (error != cudaSuccess)                                                 \
        {                                                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl;                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// NCCL error checking macro
#define NCCL_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        ncclResult_t error = call;                                                \
        if (error != ncclSuccess)                                                 \
        {                                                                         \
            std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << ncclGetErrorString(error) << std::endl;                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Warp-level shuffle reduction
__device__ size_t warpReduceMax(size_t val, float weight, float *max_weight)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other_weight = __shfl_down_sync(0xffffffff, *max_weight, offset);
        size_t other_val = __shfl_down_sync(0xffffffff, val, offset);

        if (other_weight > *max_weight)
        {
            *max_weight = other_weight;
            val = other_val;
        }
    }
    return val;
}

// Kernel for the pointing phase
__global__ void setPointersKernel(size_t *vertex_batch, size_t num_vertices_batch,
                                  size_t *offsets, size_t *edges, float *weights,
                                  size_t *pointers, size_t *mate)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices_batch)
        return;

    size_t u = vertex_batch[idx];

    // Skip if already matched
    if (mate[u] != SIZE_MAX)
        return;

    size_t best_v = SIZE_MAX;
    float best_weight = -1.0f;

    // Process neighbors
    size_t start = offsets[u];
    size_t end = offsets[u + 1];

    for (size_t e = start; e < end; ++e)
    {
        size_t v = edges[e];

        // Skip if already matched
        if (mate[v] != SIZE_MAX)
            continue;

        float w = weights[e];
        if (w > best_weight)
        {
            best_weight = w;
            best_v = v;
        }
    }

    // Set pointer to heaviest available neighbor
    pointers[u] = best_v;
    // printf("Vertex %llu points to %llu with weight %f\n", u, best_v, best_weight);
}

// Kernel for the matching phase
__global__ void setMatesKernel(size_t *vertex_batch, size_t num_vertices_batch,
                               size_t *pointers, size_t *mate)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices_batch)
        return;

    size_t u = vertex_batch[idx];

    // Skip if already matched
    if (mate[u] != SIZE_MAX)
        return;

    size_t v = pointers[u];

    // Check for mutual pointing
    if (v != SIZE_MAX && pointers[v] == u)
    {
        // Atomic operation to ensure only one thread sets the match
        if (atomicCAS(&mate[u], SIZE_MAX, v) == SIZE_MAX)
        {
            atomicCAS(&mate[v], SIZE_MAX, u);
        }
    }
}

// LD_GPU_Matcher implementation
LD_GPU_Matcher::LD_GPU_Matcher(Graph &graph, int num_gpus, int max_batches_per_device)
    : num_gpus(num_gpus), threads_per_block(256)
{

    // Initialize host arrays
    h_pointers.resize(graph.num_vertices, SIZE_MAX);
    h_mate.resize(graph.num_vertices, SIZE_MAX);

    // Set up devices
    std::cerr << "*** Requested " << num_gpus << " GPU(s), and ";
    setupDevices(num_gpus);

    std::cerr << num_gpus << " GPU(s) are available. ***" << std::endl;
    std::cout << "Using " << num_gpus << " GPU(s) for matching..." << std::endl
              << std::endl;

    // Partition graph
    graph.partitionGraph(num_gpus, graph_partitions);

    // Create batches
    createBatches(max_batches_per_device);

    // Set up NCCL for multi-GPU communication
    if (num_gpus > 1)
    {
        setupNCCL();
    }

    // Allocate memory on each device
    d_pointers.resize(num_gpus, nullptr);
    d_mate.resize(num_gpus, nullptr);

    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
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
        for (int s = 0; s < 2; ++s)
        {
            CUDA_CHECK(cudaStreamCreate(&streams[s][gpu]));
        }
    }
}

LD_GPU_Matcher::~LD_GPU_Matcher()
{
    // Clean up device memory
    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
        CUDA_CHECK(cudaSetDevice(gpu));

        if (d_pointers[gpu])
        {
            CUDA_CHECK(cudaFree(d_pointers[gpu]));
        }

        if (d_mate[gpu])
        {
            CUDA_CHECK(cudaFree(d_mate[gpu]));
        }

        // Free graph partition memory
        graph_partitions[gpu].freeDeviceMemory();

        // Destroy streams
        for (int s = 0; s < 2; ++s)
        {
            CUDA_CHECK(cudaStreamDestroy(streams[s][gpu]));
        }
    }

    // Clean up NCCL
    if (num_gpus > 1)
    {
        cleanupNCCL();
    }
}

void LD_GPU_Matcher::setupDevices(int &num_gpus)
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (num_gpus > device_count)
    {
        num_gpus = device_count;
    }
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
        
        std::cout << "// GPU " << gpu << " has " << (gpu_batches.size() - 1) << " batches. //" << std::endl;
    }
    std::cout << std::endl;
}

void LD_GPU_Matcher::setupNCCL()
{
    comms.resize(num_gpus);
    int devices[num_gpus];
    for (int i = 0; i < num_gpus; ++i)
    {
        devices[i] = i;
    }

    // Initialize NCCL communicators
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devices));
}

void LD_GPU_Matcher::cleanupNCCL()
{
    for (int i = 0; i < num_gpus; ++i)
    {
        ncclCommDestroy(comms[i]);
    }
}

bool LD_GPU_Matcher::executeIterationBatched() {
    // Allocate device memory for the has_new_matches flag
    int* d_has_new_matches;
    int has_new_matches = 0;
    CUDA_CHECK(cudaMalloc(&d_has_new_matches, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_has_new_matches, 0, sizeof(int)));
    
    // Save the old matching for comparison
    std::vector<size_t> old_mate = h_mate;
    
    // Sync host-to-device for all GPUs at the start
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMemcpy(d_mate[gpu], h_mate.data(), sizeof(size_t) * h_mate.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pointers[gpu], h_pointers.data(), sizeof(size_t) * h_pointers.size(), cudaMemcpyHostToDevice));
    }
    
    // Pointing phase
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
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
                // Correct indexing based on partition offsets
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
    }
    
    // All devices synchronize pointers via NCCL if multi-GPU
    if (num_gpus > 1) {
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            CUDA_CHECK(cudaSetDevice(gpu));
            NCCL_CHECK(ncclAllReduce(
                (const void*)d_pointers[gpu], (void*)d_pointers[gpu], 
                h_pointers.size(), ncclUint64, ncclMax, 
                comms[gpu], streams[0][gpu]));
        }
        
        // Synchronize all streams
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaStreamSynchronize(streams[0][gpu]));
        }
    }
    
    // Matching phase - each GPU works on its assigned partition
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
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
            
            // Launch kernel for matching phase
            int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
            setMatesKernel<<<blocks, threads_per_block, 0, stream>>>(
                d_vertex_batch, batch_size,
                d_pointers[gpu], d_mate[gpu]
            );
            
            // Free batch memory
            CUDA_CHECK(cudaFree(d_vertex_batch));
            
            // Synchronize stream
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    
    // Synchronize mates across GPUs if multi-GPU
    if (num_gpus > 1) {
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            CUDA_CHECK(cudaSetDevice(gpu));
            NCCL_CHECK(ncclAllReduce(
                (const void*)d_mate[gpu], (void*)d_mate[gpu], 
                h_mate.size(), ncclUint64, ncclMin, 
                comms[gpu], streams[0][gpu]));
        }
        
        // Synchronize all streams
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaStreamSynchronize(streams[0][gpu]));
        }
    }
    
    // Copy updated matching back to host from GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_mate.data(), d_mate[0], sizeof(size_t) * h_mate.size(), cudaMemcpyDeviceToHost));
    
    // Copy updated pointers back to host
    CUDA_CHECK(cudaMemcpy(h_pointers.data(), d_pointers[0], sizeof(size_t) * h_pointers.size(), cudaMemcpyDeviceToHost));
    
    // Allocate memory for old_mate on device and copy
    size_t* d_old_mate;
    CUDA_CHECK(cudaMalloc(&d_old_mate, sizeof(size_t) * h_mate.size()));
    CUDA_CHECK(cudaMemcpy(d_old_mate, old_mate.data(), sizeof(size_t) * old_mate.size(), cudaMemcpyHostToDevice));
    
    // Check for new matches
    int blocks = (h_mate.size() + threads_per_block - 1) / threads_per_block;
    checkNewMatchesKernel<<<blocks, threads_per_block>>>(
        h_mate.size(), d_mate[0], d_old_mate, d_has_new_matches
    );
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&has_new_matches, d_has_new_matches, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_old_mate));
    CUDA_CHECK(cudaFree(d_has_new_matches));
    
    std::cout << "==> New matches found: " << (has_new_matches == 1 ? "yes" : "no") << std::endl;
    
    return has_new_matches == 1;
}

void LD_GPU_Matcher::computeMatching() {
    int iteration = 0;
    bool continue_matching = true;
    
    // Initialize pointers to SIZE_MAX
    std::fill(h_pointers.begin(), h_pointers.end(), SIZE_MAX);
    
    while (continue_matching) {
        std::cout << "  [Starting iteration " << iteration << "]" << std::endl;
        
        continue_matching = executeIterationBatched();
        
        if (!continue_matching) {
            std::cout << "==> No new matches found in iteration " << iteration << std::endl << std::endl;
            break;
        }
        
        std::cout << std::endl;
        iteration++;
    }
    
    // Count matches
    size_t num_matches = 0;
    for (size_t i = 0; i < h_mate.size(); ++i) {
        if (h_mate[i] != SIZE_MAX && i < h_mate[i]) {
            num_matches++;
        }
    }
    
    std::cout << ">>>> Matching completed. <<<<" << std::endl;
    std::cout << "\n\n# Final Results:" << std::endl;
    std::cout << "- Matching completed in " << iteration << " iterations." << std::endl;
    std::cout << "- Found " << num_matches << " matched pairs." << std::endl;
}

const std::vector<size_t> &LD_GPU_Matcher::getMatching() const
{
    return h_mate;
}
