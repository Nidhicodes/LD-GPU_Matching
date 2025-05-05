import numpy as np
import sys

def generate_queen_like_graph(n, output_file):
    """
    Generate a symmetric Queen-like graph on an n×n grid.
    Each node connects based on queen moves. Undirected edges with symmetric weights.
    """
    edge_weights = {}  # key: frozenset({a, b}), value: weight

    def add_edge(a, b):
        key = frozenset({a, b})
        if a != b and key not in edge_weights:
            weight = np.random.uniform(0.1, 1.0)
            edge_weights[key] = weight

    for i in range(n):
        for j in range(n):
            vertex_id = i * n + j

            # Horizontal
            for k in range(n):
                if k != j:
                    target = i * n + k
                    add_edge(vertex_id, target)

            # Vertical
            for k in range(n):
                if k != i:
                    target = k * n + j
                    add_edge(vertex_id, target)

            # Diagonal and anti-diagonal
            for k in range(1, n):
                moves = [
                    (i + k, j + k),  # down-right
                    (i - k, j - k),  # up-left
                    (i + k, j - k),  # down-left
                    (i - k, j + k),  # up-right
                ]
                for x, y in moves:
                    if 0 <= x < n and 0 <= y < n:
                        target = x * n + y
                        add_edge(vertex_id, target)

    node_edges = {i: [] for i in range(n * n)}
    for key, weight in edge_weights.items():
        a, b = tuple(key)
        node_edges[a].append((b, weight))
        node_edges[b].append((a, weight))

    with open(output_file, 'w') as f:
        f.write(f"# Symmetric Queen-like graph on {n}x{n} grid\n")
        f.write("# Format: source target weight\n")

        for src in range(n * n):
            for tgt, weight in sorted(node_edges[src], key=lambda x: x[0]):
                f.write(f"{src} {tgt} {weight:.6f}\n")

def generate_mycielskian_graph(n, output_file):
    """
    Generate the Mycielskian of a simple path graph of length n.
    The Mycielskian graph construction increases chromatic number without increasing the clique size.
    """
    def random_weight():
        return np.random.uniform(0.1, 1.0)

    V = list(range(n))
    U = list(range(n, 2 * n))
    z = 2 * n

    edges = {}

    # Edges of original path graph
    for i in range(n - 1):
        a, b = i, i + 1
        w = random_weight()
        edges[frozenset({a, b})] = w

    # For each edge (i, j) in original graph, connect i with j and u_i with u_j
    for key in list(edges.keys()):
        a, b = tuple(key)
        w = edges[key]
        edges[frozenset({a, b})] = w  # Already added
        edges[frozenset({a + n, b + n})] = w  # u_i <-> u_j

    # Connect each v_i to u_i
    for i in range(n):
        w = random_weight()
        edges[frozenset({i, i + n})] = w

    # Connect z to all u_i
    for i in range(n, 2 * n):
        w = random_weight()
        edges[frozenset({z, i})] = w

    total_nodes = 2 * n + 1
    node_edges = {i: [] for i in range(total_nodes)}
    for key, weight in edges.items():
        a, b = tuple(key)
        node_edges[a].append((b, weight))
        node_edges[b].append((a, weight))

    with open(output_file, 'w') as f:
        f.write(f"# Mycielskian graph based on path of size {n}\n")
        f.write("# Format: source target weight\n")
        for src in range(total_nodes):
            for tgt, weight in sorted(node_edges[src], key=lambda x: x[0]):
                f.write(f"{src} {tgt} {weight:.6f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <graph_type> <param> <output_file>")
        print("  graph_type: 'queen' or 'mycielskian'")
        print("  param: size parameter (n for grid size or path length)")
        print("  output_file: path to output file")
        sys.exit(1)

    graph_type = sys.argv[1].lower()
    param = int(sys.argv[2])
    output_file = sys.argv[3]

    if graph_type == "queen":
        generate_queen_like_graph(param, output_file)
        print(f"Generated Queen-like graph with {param}x{param} grid to {output_file}")
    elif graph_type == "mycielskian":
        generate_mycielskian_graph(param, output_file)
        print(f"Generated Mycielskian graph from path of length {param} to {output_file}")
    else:
        print(f"Unknown graph type: {graph_type}")
        sys.exit(1)
