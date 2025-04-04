#ifndef GRAPH_CUH
#define GRAPH_CUH

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

// CSR format graph structure
struct Graph {
    // Host data
    size_t num_vertices;
    size_t num_edges;
    std::vector<size_t> offsets;   // CSR offsets
    std::vector<size_t> edges;     // Edge targets
    std::vector<float> weights;    // Edge weights

    // Device data
    size_t* d_offsets;
    size_t* d_edges;
    float* d_weights;

    // Constructor
    Graph() : num_vertices(0), num_edges(0), d_offsets(nullptr), d_edges(nullptr), d_weights(nullptr) {}

    // Load graph from file (simple format: source target weight)
    void loadFromFile(const std::string& filename);
    
    // Allocate device memory and copy graph to device
    void copyToDevice();
    
    // Free device memory
    void freeDeviceMemory();
    
    // Partition graph for multi-GPU processing
    void partitionGraph(int num_gpus, std::vector<Graph>& partitions);
};

#endif // GRAPH_CUH
