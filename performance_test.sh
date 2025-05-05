#!/bin/bash

# Create directories if they don't exist
mkdir -p ../data
mkdir -p results

# Set number of runs per test
RUNS=3

# Function to generate test graph if it doesn't exist
generate_graph() {
    local size=$1
    local density=$2
    local outfile="../data/graph_${size}.txt"
    
    if [ ! -f "$outfile" ]; then
        echo "Generating graph with $size vertices and density $density..."
        python generate_test_graph.py "$size" "$density" "$outfile"
    else
        echo "Graph $outfile already exists, skipping generation."
    fi
    
    return 0
}

# Generate test graphs with different sizes and densities
generate_graph 1000 0.01
generate_graph 5000 0.005
generate_graph 10000 0.001
generate_graph 50000 0.0005
generate_graph 100000 0.0001

# Function to run benchmark with specified parameters
run_benchmark() {
    local graph=$1
    local gpus=$2
    local batch_size=${3:-2}  # Default batch size is 2
    local outfile="results/$(basename $graph)_${gpus}gpu_batch${batch_size}.txt"
    
    echo "  Running with $gpus GPU(s), batch size $batch_size..."
    ./ld_gpu_matching "$graph" "$gpus" "$batch_size" > "$outfile" 2>&1
    
    # Extract execution time
    local time=$(grep "Total execution time" "$outfile" | awk '{print $5}')
    local matches=$(grep "Found" "$outfile" | awk '{print $3}')
    
    echo "    Execution time: ${time:-N/A}s, Matches: ${matches:-N/A}"
    
    return 0
}

# Run tests with varying number of GPUs and batch sizes
echo "Starting performance tests..."

# Get number of available GPUs
MAX_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $MAX_GPUS GPU(s) in the system."

# Determine which GPUs to test with
if [ $MAX_GPUS -eq 0 ]; then
    echo "Error: No GPUs detected! Exiting."
    exit 1
fi

# Define GPU counts to test
GPU_COUNTS=(1)
if [ $MAX_GPUS -ge 2 ]; then
    GPU_COUNTS+=(2)
fi
if [ $MAX_GPUS -ge 4 ]; then
    GPU_COUNTS+=(4)
fi
if [ $MAX_GPUS -ge 8 ]; then
    GPU_COUNTS+=(8)
fi

# Define batch sizes to test
BATCH_SIZES=(1 2 4)

for graph in ../data/graph_*.txt; do  
    echo "Testing $graph"
    
    for gpus in "${GPU_COUNTS[@]}"; do
        echo "With $gpus GPU(s):"
        
        for batch in "${BATCH_SIZES[@]}"; do
            # Run multiple times for better statistics
            for ((i=1; i<=$RUNS; i++)); do
                echo "  Run $i/$RUNS:"
                run_benchmark "$graph" "$gpus" "$batch"
            done
        done
    done
done

# Generate summary file
echo "Preparing summary of results..."
{
    echo "Graph,GPUs,BatchSize,AverageTime,AverageMatches"
    
    for graph in ../data/graph_*.txt; do
        graph_base=$(basename "$graph")
        
        for gpus in "${GPU_COUNTS[@]}"; do
            for batch in "${BATCH_SIZES[@]}"; do
                # Calculate averages from all runs
                times=()
                matches=()
                
                for f in "results/${graph_base}_${gpus}gpu_batch${batch}.txt"; do
                    if [ -f "$f" ]; then
                        time=$(grep "Total execution time" "$f" | awk '{print $5}')
                        match=$(grep "Found" "$f" | awk '{print $3}')
                        
                        if [ ! -z "$time" ]; then
                            times+=("$time")
                        fi
                        
                        if [ ! -z "$match" ]; then
                            matches+=("$match")
                        fi
                    fi
                done
                
                # Calculate average time
                total_time=0
                for t in "${times[@]}"; do
                    total_time=$(echo "$total_time + $t" | bc)
                done
                
                if [ ${#times[@]} -gt 0 ]; then
                    avg_time=$(echo "scale=4; $total_time / ${#times[@]}" | bc)
                else
                    avg_time="N/A"
                fi
                
                # Calculate average matches
                total_matches=0
                for m in "${matches[@]}"; do
                    total_matches=$((total_matches + m))
                done
                
                if [ ${#matches[@]} -gt 0 ]; then
                    avg_matches=$((total_matches / ${#matches[@]}))
                else
                    avg_matches="N/A"
                fi
                
                echo "$graph_base,$gpus,$batch,$avg_time,$avg_matches"
            done
        done
    done
} > results/summary.csv

echo "Performance tests completed. Results are in the 'results' directory."
echo "Summary available in results/summary.csv"