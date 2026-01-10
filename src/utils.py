import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.lines as mlines
from src.loading_data import load_graph  # Importamos la función de lectura

def loading_graphs(path, size=""):
    if os.path.exists(path):
        graph = load_graph(path)
        if graph:
            print(f"{size} Graph Loaded")
            print(f"Nodes: {graph.number_of_nodes()}")
            print(f"Edges: {graph.number_of_edges()}")
        return graph
    else:
        print(f"File {path} not found.")
        return None

def plot_graph(graph, title="Graph"):
    plt.figure(figsize=(10, 8))
    # Usamos spring_layout con seed fija para consistencia
    pos = nx.spring_layout(graph, seed=42) 
    nx.draw(graph, pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            node_size=500,
            font_size=10)
    plt.title(title)
    plt.show()

def plot_colored_graph(G, individual, title="GCP Solution"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Preparar colores
    unique_genes = np.unique(individual.genes)
    cmap = plt.cm.rainbow
    norm = plt.Normalize(vmin=min(individual.genes), vmax=max(individual.genes))
    
    # Asignar color a cada nodo. OJO: Los nodos de networkx empiezan en 1
    # pero el array genes empieza en índice 0. Hay que alinear.
    node_colors = [individual.genes[n-1] for n in G.nodes()]

    nx.draw(G, pos,
            with_labels=True,
            node_color=node_colors,
            cmap=cmap,
            edge_color='black'
            node_size=600,
            font_size=12,
            width=1.5,
            alpha=0.9)
    
    # Crear Leyenda manual
    legend_handles = []
    for gene in sorted(unique_genes):
        color = cmap(norm(gene))
        line = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                             markersize=10, label=f'Color ID: {gene}')
        legend_handles.append(line)

    plt.legend(handles=legend_handles, title="Colors", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f"{title}\nConflicts: {individual.conflicts} | Total Colors: {len(unique_genes)}")
    plt.tight_layout()
    plt.show()