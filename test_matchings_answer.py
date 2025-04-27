import networkx as nx

def find_max_weighted_matching(input_file):
    G = nx.Graph()
    
    # Read the input file and add edges to the graph
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            u, v, w = line.strip().split()
            G.add_edge(int(u), int(v), weight=float(w))
    
    # Find the maximum weight matching using NetworkX's built-in function
    matching = nx.max_weight_matching(G, maxcardinality=True, weight='weight')
    
    # Output how many matchings were found
    return len(matching)

if __name__ == "__main__":
    # Provide the input file
    input_file = input("Enter the path of the input file: ")
    
    match_count = find_max_weighted_matching(input_file)
    print(f"Number of matchings found: {match_count}")
