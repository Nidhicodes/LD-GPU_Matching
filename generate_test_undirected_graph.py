import numpy as np
import sys

def generate_random_graph(num_vertices, edge_probability, output_file):
    edges = []

    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if np.random.random() < edge_probability:
                weight = np.random.uniform(0.1, 1.0)  # Random float weight between 0.1 and 1.0
                edges.append((u, v, weight))
                edges.append((v, u, weight))  # Add reverse edge

    # Sort edges by src, then dst
    edges.sort()

    with open(output_file, 'w') as f:
        f.write(f"# Random undirected graph with {num_vertices} vertices\n")
        f.write(f"# Format: source target weight\n")
        for u, v, weight in edges:
            f.write(f"{u} {v} {weight:.6f}\n")  # 6 digits after decimal

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <num_vertices> <edge_probability> <output_file>")
        sys.exit(1)

    num_vertices = int(sys.argv[1])
    edge_probability = float(sys.argv[2])
    output_file = sys.argv[3]

    generate_random_graph(num_vertices, edge_probability, output_file)
    print(f"Generated random undirected graph with {num_vertices} vertices to {output_file}")
