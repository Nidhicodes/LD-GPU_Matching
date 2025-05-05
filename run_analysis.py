import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_result_file(filename):
    """Parse the output file to extract relevant metrics"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
            # Extract execution time
            time_match = re.search(r"Total execution time: ([0-9.]+) seconds", content)
            execution_time = float(time_match.group(1)) if time_match else None
            
            # Extract number of iterations
            iter_match = re.search(r"Matching completed in ([0-9]+) iterations", content)
            iterations = int(iter_match.group(1)) if iter_match else None
            
            # Extract number of matched pairs
            pairs_match = re.search(r"Found ([0-9]+) matched pairs", content)
            matched_pairs = int(pairs_match.group(1)) if pairs_match else None
            
            # Extract vertex and edge counts
            vertices_match = re.search(r"Graph loaded: ([0-9]+) vertices", content)
            vertices = int(vertices_match.group(1)) if vertices_match else None
            
            edges_match = re.search(r"([0-9]+) edges", content)
            edges = int(edges_match.group(1)) if edges_match else None
            
            return {
                "execution_time": execution_time,
                "iterations": iterations,
                "matched_pairs": matched_pairs,
                "vertices": vertices,
                "edges": edges
            }
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def calculate_mmeps(results):
    """Calculate Mega-Matching Edges per Second (MMEPS)"""
    if not results or not all(key in results for key in ["matched_pairs", "execution_time", "iterations"]):
        return None
    
    # MMEPS = (matched pairs * 2) / (execution time * 10^6)
    # The *2 is because each pair represents 2 edges
    return (results["matched_pairs"] * 2) / (results["execution_time"] * 1e6)

def analyze_results(results_dir):
    """Analyze all result files in the given directory"""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(results_dir, filename)
            base_name = filename[:-4]  # Remove .txt
            
            parsed_results = parse_result_file(filepath)
            if parsed_results:
                parsed_results["mmeps"] = calculate_mmeps(parsed_results)
                results[base_name] = parsed_results
    
    return results

def print_report(results):
    """Print a formatted report of the results"""
    print("\n===== LD-GPU MATCHING ALGORITHM RESULTS =====\n")
    
    # Group results by graph
    graph_results = {}
    for key, data in results.items():
        graph_name = key.split('_')[0] + '_' + key.split('_')[1]  # Extract graph name without GPU count
        gpu_count = int(key.split('_')[-1].replace('gpu', ''))
        
        if graph_name not in graph_results:
            graph_results[graph_name] = {}
        
        graph_results[graph_name][gpu_count] = data
    
    # Print results by graph
    for graph_name, gpu_data in graph_results.items():
        print(f"Graph: {graph_name}")
        print(f"  Vertices: {gpu_data[1]['vertices']}")
        print(f"  Edges: {gpu_data[1]['edges']}")
        print("  Performance:")
        
        for gpu_count, data in sorted(gpu_data.items()):
            print(f"    {gpu_count} GPU{'s' if gpu_count > 1 else ''}:")
            print(f"      Execution Time: {data['execution_time']:.6f} seconds")
            print(f"      Iterations: {data['iterations']}")
            print(f"      Matched Pairs: {data['matched_pairs']}")
            if data['mmeps'] is not None:
                print(f"      MMEPS: {data['mmeps']:.2f}")
        
        print()
    
    # Compare with paper results
    print("===== COMPARISON WITH PAPER RESULTS =====\n")
    print("Note: The synthetic graphs generated are not identical to those in the paper,")
    print("so direct comparisons may not be accurate. The following are approximate comparisons:\n")
    
    paper_results = {
        "queen": {"execution_time": 0.027},
        "mycielskian": {"execution_time": 0.019}
    }
    
    for graph_type in paper_results:
        matching_graphs = [g for g in graph_results if graph_type in g]
        if matching_graphs:
            graph_name = matching_graphs[0]
            our_time = graph_results[graph_name][1]['execution_time']
            paper_time = paper_results[graph_type]["execution_time"]
            
            print(f"{graph_type.capitalize()} graph:")
            print(f"  Paper result (LD-GPU): {paper_time:.3f} seconds")
            print(f"  Our result (LD-GPU): {our_time:.3f} seconds")
            print(f"  Ratio: {our_time/paper_time:.2f}x\n")

def plot_results(results):
    """Create performance visualization plots"""
    # Group results by graph
    graph_results = {}
    for key, data in results.items():
        graph_name = key.split('_')[0] + '_' + key.split('_')[1]  # Extract graph name without GPU count
        gpu_count = int(key.split('_')[-1].replace('gpu', ''))
        
        if graph_name not in graph_results:
            graph_results[graph_name] = {}
        
        graph_results[graph_name][gpu_count] = data
    
    # Execution time comparison
    plt.figure(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set positions of bars on X axis
    r1 = np.arange(len(graph_results))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    single_gpu_times = [graph_results[g][1]['execution_time'] if 1 in graph_results[g] else 0 for g in graph_results]
    multi_gpu_times = [graph_results[g][2]['execution_time'] if 2 in graph_results[g] else 0 for g in graph_results]
    
    plt.bar(r1, single_gpu_times, width=bar_width, label='1 GPU', color='skyblue')
    
    # Only plot multi-GPU if we have data
    if any(multi_gpu_times):
        plt.bar(r2, multi_gpu_times, width=bar_width, label='2 GPUs', color='darkblue')
    
    # Add labels and title
    plt.xlabel('Graph')
    plt.ylabel('Execution Time (seconds)')
    plt.title('LD-GPU Execution Time by Graph and GPU Count')
    plt.xticks([r + bar_width/2 for r in range(len(graph_results))], list(graph_results.keys()))
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png')
    print("Plot saved as 'execution_time_comparison.png'")

if __name__ == "__main__":
    results_dir = "build/results"
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        exit(1)
    
    results = analyze_results(results_dir)
    print_report(results)
    plot_results(results)