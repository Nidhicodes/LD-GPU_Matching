import random
import os

# Define the directory to save the graph
output_directory = "/home/user/LD-GPU_Matching/data"

# Ensure the directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Parameters for the graph
num_vertices = 1000
num_edges = 5000
max_degree = 50
min_weight = 0
max_weight = 1

# Generate random edges with weights
edges = set()
while len(edges) < num_edges:
    u = random.randint(0, num_vertices - 1)
    v = random.randint(0, num_vertices - 1)
    if u != v:  # avoid self-loops
        weight = round(random.uniform(min_weight, max_weight), 3)
        edges.add((u, v, weight))

# File path for saving the graph
graph_file_path = os.path.join(output_directory, "graph_1k.txt")

# Write the graph to a text file in edge list format
with open(graph_file_path, "w") as file:
    for u, v, weight in edges:
        file.write(f"{u} {v} {weight}\n")

print(f"Graph with {num_vertices} vertices and {num_edges} edges saved to '{graph_file_path}'")
