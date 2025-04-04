#!/bin/bash

mkdir -p results

python generate_test_graph.py 1000 0.01 ../data/graph_1k.txt
python generate_test_graph.py 5000 0.005 ../data/graph_5k.txt
python generate_test_graph.py 10000 0.001 ../data/graph_10k.txt
python generate_test_graph.py 50000 0.0005 ../data/graph_50k.txt

# Run tests with varying number of GPUs
for graph in ../data/graph_*.txt; do  
    echo "Testing $graph"
    
    for gpus in 1 2; do
        if [ $gpus -le $(nvidia-smi --list-gpus | wc -l) ]; then
            echo "  With $gpus GPU(s)..."
            ./ld_gpu_matching "$graph" $gpus 2 > "results/$(basename $graph)_${gpus}gpu.txt"
        fi
    done
done

echo "Performance tests completed. Results are in the 'results' directory."

