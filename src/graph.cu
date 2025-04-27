#include "../include/graph.cuh"
#include <sstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>

void Graph::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '#') break;
    }

    //to store the edge (source, destination)
    std::vector<std::pair<size_t, size_t>> edge_list;
    std::vector<float> weight_list;  // Separate vector to store weights
    size_t max_vertex_id = 0;

    do {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        size_t src, dst;
        float weight;
        
        if (!(iss >> src >> dst >> weight)) {
            // Try with default weight of 1.0
            iss.clear();
            iss.str(line);
            if (!(iss >> src >> dst)) {
                continue;
            }
            weight = 1.0f;
        }
        
        edge_list.emplace_back(src, dst);
        weight_list.push_back(weight);
        max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
    } while (std::getline(file, line));
    
    num_vertices = max_vertex_id + 1;  // +1 because vertices are 0-indexed
    num_edges = edge_list.size();
    
    // Sort edges by source vertex
    std::sort(edge_list.begin(), edge_list.end());
    
    // Build CSR format
    offsets.resize(num_vertices + 1, 0);
    edges.resize(num_edges);
    weights.resize(num_edges);
    
    for (size_t i = 0; i < num_edges; ++i) {
        size_t src = edge_list[i].first;   // Access the first element (src)
        edges[i] = edge_list[i].second;    // Access the second element (dst)
        weights[i] = weight_list[i];       // Access the weight from the weight_list
        offsets[src + 1]++;
    }
    
    // Convert counts to offsets
    for (size_t i = 1; i <= num_vertices; ++i) {
        offsets[i] += offsets[i - 1];
    }
    
    std::cout << "\n {Graph loaded: " << num_vertices << " vertices, " << num_edges << " edges}" << std::endl<< std::endl;
}

void Graph::copyToDevice() {
    cudaError_t error;

    // Allocate memory on device
    error = cudaMalloc((void**)&d_offsets, sizeof(size_t) * (num_vertices + 1));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for offsets: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_edges, sizeof(size_t) * num_edges);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for edges: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_offsets);
        d_offsets = nullptr;
        return;
    }

    error = cudaMalloc((void**)&d_weights, sizeof(float) * num_edges);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for weights: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_offsets);
        cudaFree(d_edges);
        d_offsets = nullptr;
        d_edges = nullptr;
        return;
    }

    // Copy data to device
    error = cudaMemcpy(d_offsets, offsets.data(), sizeof(size_t) * (num_vertices + 1), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy offsets to device: " << cudaGetErrorString(error) << std::endl;
        freeDeviceMemory();
        return;
    }

    error = cudaMemcpy(d_edges, edges.data(), sizeof(size_t) * num_edges, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy edges to device: " << cudaGetErrorString(error) << std::endl;
        freeDeviceMemory();
        return;
    }

    error = cudaMemcpy(d_weights, weights.data(), sizeof(float) * num_edges, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy weights to device: " << cudaGetErrorString(error) << std::endl;
        freeDeviceMemory();
        return;
    }
}

void Graph::freeDeviceMemory() {
    if (d_offsets) {
        cudaFree(d_offsets);
        d_offsets = nullptr;
    }

    if (d_edges) {
        cudaFree(d_edges);
        d_edges = nullptr;
    }

    if (d_weights) {
        cudaFree(d_weights);
        d_weights = nullptr;
    }
}

void Graph::partitionGraph(int num_gpus, std::vector<Graph>& partitions) {
    partitions.clear();
    partitions.resize(num_gpus);

    // Simple edge-based partitioning
    size_t edges_per_partition = (num_edges + num_gpus - 1) / num_gpus;

    // For each partition, find the vertex range
    std::vector<size_t> partition_start_vertex(num_gpus, 0);
    std::vector<size_t> partition_end_vertex(num_gpus, 0);

    for (int p = 0; p < num_gpus; ++p) {
        size_t target_edge_count = (p + 1) * edges_per_partition;
        if (p == num_gpus - 1) target_edge_count = num_edges;

        size_t v = (p == 0) ? 0 : partition_end_vertex[p - 1];
        partition_start_vertex[p] = v;

        while (v < num_vertices && offsets[v] < target_edge_count) {
            v++;
        }

        partition_end_vertex[p] = v;
    }

    // Create the partitions
    for (int p = 0; p < num_gpus; ++p) {
        Graph& partition = partitions[p];
        size_t start_v = partition_start_vertex[p];
        size_t end_v = partition_end_vertex[p];

        partition.num_vertices = end_v - start_v;

        // Copy the offsets for the partition's vertices
        partition.offsets.resize(partition.num_vertices + 1);

        size_t offset_base = offsets[start_v];
        for (size_t i = 0; i <= partition.num_vertices; ++i) {
            partition.offsets[i] = offsets[start_v + i] - offset_base;
        }

        // Copy the edges and weights for the partition
        size_t start_e = offsets[start_v];
        size_t end_e = offsets[end_v];
        partition.num_edges = end_e - start_e;

        partition.edges.resize(partition.num_edges);
        partition.weights.resize(partition.num_edges);

        for (size_t i = 0; i < partition.num_edges; ++i) {
            partition.edges[i] = edges[start_e + i];
            partition.weights[i] = weights[start_e + i];
        }

        std::cout << ">> Partition " << p << ": " << partition.num_vertices << " vertices, "
                  << partition.num_edges << " edges. <<" << std::endl;
    }
    std::cout << std::endl;

}

