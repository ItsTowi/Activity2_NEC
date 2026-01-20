import numpy as np
import random
from src.individual import Individual

def tournament_selection(population, k=3):
    sample = random.sample(population, k)
    return min(sample, key=lambda ind: ind.fitness)

def roulette_wheel_selection(population):
    fitness_values = [1.0 / (ind.fitness + 1e-6) for ind in population]
    total_fit = sum(fitness_values)
    probs = [f / total_fit for f in fitness_values]
    return np.random.choice(population, p=probs)

def one_point_crossover(p1, p2, num_nodes, max_colors):
    point = random.randint(1, num_nodes - 1)
    h1 = Individual(num_nodes, max_colors)
    h2 = Individual(num_nodes, max_colors)
    
    h1.genes = np.concatenate((p1.genes[:point], p2.genes[point:]))
    h2.genes = np.concatenate((p2.genes[:point], p1.genes[point:]))
    
    return h1, h2

def uniform_crossover(p1, p2, num_nodes, max_colors):
    h1 = Individual(num_nodes, max_colors)
    h2 = Individual(num_nodes, max_colors)
    
    mask = np.random.rand(num_nodes) < 0.5
    h1.genes = np.where(mask, p1.genes, p2.genes)
    h2.genes = np.where(mask, p2.genes, p1.genes)
    
    return h1, h2

def single_gene_mutation(ind, mutation_rate, max_colors):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(ind.genes) - 1)
        ind.genes[idx] = random.randint(0, max_colors - 1)
    return ind

def independent_gene_mutation(ind, mutation_rate, max_colors, edges_array):
    genes = ind.genes
    u_vals = genes[edges_array[:, 0]]
    v_vals = genes[edges_array[:, 1]]
    
    conflict_mask = (u_vals == v_vals)
    
    if np.any(conflict_mask):
        conflict_edges = edges_array[conflict_mask]
        
        conflict_nodes = np.unique(conflict_edges)
        
        for node_idx in conflict_nodes:
            if random.random() < 0.5:
                 ind.genes[node_idx] = random.randint(0, max_colors - 1)
    
    else:
        if random.random() < mutation_rate:
            idx = random.randint(0, len(ind.genes) - 1)
            ind.genes[idx] = random.randint(0, max_colors - 1)
            
    return ind