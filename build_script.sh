#!/bin/bash

# Create directories
mkdir -p build
mkdir -p data

# Generate test graphs similar to those in the paper
echo "Generating synthetic test graphs..."
python3 scripts/synthetic_graph_generator.py queen 64 queen_64.txt
python3 scripts/synthetic_graph_generator.py mycielskian 10 mycielskian_10.txt

# Build the project
echo "Building LD-GPU project..."
cd build
cmake ..
make

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Create a directory for results
    mkdir -p results
    
    # Run the algorithm on the test graphs with 1 and 2 GPUs
    cd ..
    
    echo "Running tests with Queen-like graph..."
    available_gpus=$(nvidia-smi --list-gpus | wc -l)
    
    for gpus in 1 $([ $available_gpus -ge 2 ] && echo "2"); do
        echo "  With $gpus GPU(s)..."
        build/ld_gpu_matching queen_64.txt $gpus 2 > build/results/queen_64_${gpus}gpu.txt
    done
    
    echo "Running tests with Mycielskian-like graph..."
    for gpus in 1 $([ $available_gpus -ge 2 ] && echo "2"); do
        echo "  With $gpus GPU(s)..."
        build/ld_gpu_matching mycielskian_10.txt $gpus 2 > build/results/mycielskian_10_${gpus}gpu.txt
    done
    
    echo "Tests completed! Results are in build/results directory."
else
    echo "Build failed!"
fi