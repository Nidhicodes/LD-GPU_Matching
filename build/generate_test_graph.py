import numpy as np
import sys

def generate_random_graph(num_vertices, edge_probability, output_file):
    with open(output_file, 'w') as f:
        f.write(f"# Random graph with {num_vertices} vertices\n")
        f.write(f"# Format: source target weight\n")
        
        for u in range(num_vertices):
            # Generate random edges
            for v in range(u+1, num_vertices):  # Undirected graph, so only need one direction
                if np.random.random() < edge_probability:
                    weight = np.random.uniform(0.1, 1.0)  # Random weight between 0.1 and 1.0
                    f.write(f"{u} {v} {weight:.6f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <num_vertices> <edge_probability> <output_file>")
        sys.exit(1)
    
    num_vertices = int(sys.argv[1])
    edge_probability = float(sys.argv[2])
    output_file = sys.argv[3]
    
    generate_random_graph(num_vertices, edge_probability, output_file)
    print(f"Generated random graph with {num_vertices} vertices to {output_file}")
