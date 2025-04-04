#include <iostream>
#include <string>
#include <chrono>
#include "../include/graph.cuh"
#include "../include/ld_gpu.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <num_gpus> [max_batches_per_gpu]" << std::endl;
        return 1;
    }
    
    std::string graph_file = argv[1];
    int num_gpus = std::stoi(argv[2]);
    int max_batches = (argc >= 4) ? std::stoi(argv[3]) : 2;
    
    // Load graph
    Graph graph;
    graph.loadFromFile(graph_file);
    
    // Create matcher
    auto start = std::chrono::high_resolution_clock::now();
    
    LD_GPU_Matcher matcher(graph, num_gpus, max_batches);
    matcher.computeMatching();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}
