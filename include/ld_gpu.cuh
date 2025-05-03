#ifndef LD_GPU_CUH
#define LD_GPU_CUH

#include "graph.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>

class LD_GPU_Matcher {
private:
    int num_gpus;
    std::vector<Graph> graph_partitions;
    std::vector<size_t*> d_pointers;  // Device pointers arrays
    std::vector<size_t*> d_mate;      // Device mate arrays
    std::vector<cudaStream_t> streams[2]; // Two streams per device
    std::vector<ncclComm_t> comms;    // NCCL communicators
    
    // Batch information
    std::vector<std::vector<size_t>> batch_offsets; // Batch vertex ranges
    std::vector<size_t> gpu_vertex_offsets;  // Global vertex ID offset for each GPU
    
    // Host arrays for all vertices
    std::vector<size_t> h_pointers;
    std::vector<size_t> h_mate;
    
    // Kernel launch parameters
    int threads_per_block;
    
    void setupDevices(int& num_gpus);
    void createBatches(int max_batches_per_device);
    void setupNCCL();
    void cleanupNCCL();
    bool executeIterationBatched();
    void debugMatchingState();
    void validateMatching();
    
public:
    LD_GPU_Matcher(Graph& graph, int num_gpus, int max_batches_per_device = 2);
    ~LD_GPU_Matcher();
    
    void computeMatching();
    
    const std::vector<size_t>& getMatching() const;
};

#endif