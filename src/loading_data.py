import networkx as nx

def load_graph(path):
    G = nx.Graph()
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('c'): continue
                parts = line.split()
                if not parts: continue
                
                if parts[0] == 'p':
                    num_nodes = int(parts[2])
                    G.add_nodes_from(range(1, num_nodes + 1))
                elif parts[0] == 'e':
                    u, v = int(parts[1]), int(parts[2])
                    G.add_edge(u, v)
        return G
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {path}")
        return None