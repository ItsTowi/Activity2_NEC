import numpy as np

class Individual:
    def __init__(self, num_nodes, max_colors):
        self.genes = np.random.randint(0, max_colors, size=num_nodes)
        self.fitness = float('inf')
        self.conflicts = 0