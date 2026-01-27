import numpy as np
CONFLICT_WEIGHT = 1028

class GraphColoringSolver:
    def __init__(self, graph, n_colors, pop_size=100, mutation_rate=0.05, elitism_count=2, w_conflict = None):
        self.n_nodes = graph.number_of_nodes()
        self.n_colors = n_colors
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        if (w_conflict != None):
            self.conflict_weight = w_conflict
        else:
            self.conflict_weight = CONFLICT_WEIGHT

        self.edges = np.array(list(graph.edges()))
        if self.edges.min() > 0: self.edges -= 1
            
        self.population = np.random.randint(0, self.n_colors, size=(self.pop_size, self.n_nodes))
        self.fitness_scores = np.zeros(self.pop_size)

    def evaluate(self):
        """Calcula conflictos y suma el número de colores usados."""
        
        # --- PARTE 1: Calcular Conflictos (Tu código original) ---
        colors_u = self.population[:, self.edges[:, 0]]
        colors_v = self.population[:, self.edges[:, 1]]
        conflicts = np.sum(colors_u == colors_v, axis=1)
        
        # --- PARTE 2: Calcular Colores Únicos (El método rápido) ---
        # 1. Creamos una COPIA ordenada para contar (no altera self.population)
        sorted_pop = np.sort(self.population, axis=1)
        
        # 2. Vemos dónde hay cambios de valor (diferencias con el vecino)
        diffs = sorted_pop[:, 1:] != sorted_pop[:, :-1]
        
        # 3. Sumamos diferencias + 1 (el primer color siempre cuenta)
        n_colors_used = np.sum(diffs, axis=1) + 1
        
        # --- PARTE 3: Combinar (Fórmula Final) ---
        # Vector (100,) + Vector (100,) = Vector (100,)
        self.fitness_scores = (conflicts * self.conflict_weight) + n_colors_used
        
        return self.fitness_scores

    def select_tournament(self, k=4):
        contenders = np.random.randint(0, self.pop_size, size=(self.pop_size, k))
        fitness_vals = self.fitness_scores[contenders]
        winners_idx = np.argmin(fitness_vals, axis=1)
        rows = np.arange(self.pop_size)
        return self.population[contenders[rows, winners_idx]].copy()

    def select_roulette(self):
        inv_fitness = 1.0 / (self.fitness_scores + 1e-6)
        probs = inv_fitness / np.sum(inv_fitness)
        indices = np.random.choice(self.pop_size, size=self.pop_size, p=probs)
        return self.population[indices].copy()

    def crossover_uniform(self, parents):
        p1 = parents
        p2 = np.roll(parents, 1, axis=0)
        mask = np.random.rand(self.pop_size, self.n_nodes) < 0.5
        return np.where(mask, p1, p2)

    def crossover_one_point(self, parents):
        p1 = parents
        p2 = np.roll(parents, 1, axis=0)
        cuts = np.random.randint(1, self.n_nodes, size=(self.pop_size, 1))
        indices = np.arange(self.n_nodes).reshape(1, -1)
        mask = indices < cuts
        return np.where(mask, p1, p2)

    def mutate_smart(self, offspring):
        for i in range(self.pop_size):
            genes = offspring[i]
            u = genes[self.edges[:, 0]]
            v = genes[self.edges[:, 1]]
            mask = (u == v)
            if np.any(mask):
                bad_nodes = np.unique(self.edges[mask])
                mut_mask = np.random.rand(len(bad_nodes)) < 0.5
                nodes_to_change = bad_nodes[mut_mask]
                new_cols = np.random.randint(0, self.n_colors, size=len(nodes_to_change))
                genes[nodes_to_change] = new_cols
            if np.random.rand() < self.mutation_rate:
                idx = np.random.randint(0, self.n_nodes)
                genes[idx] = np.random.randint(0, self.n_colors)
            offspring[i] = genes
        return offspring

    def mutate_random(self, offspring):
        mask = np.random.rand(self.pop_size, self.n_nodes) < self.mutation_rate
        new_vals = np.random.randint(0, self.n_colors, size=np.sum(mask))
        offspring[mask] = new_vals
        return offspring

    # --- BUCLE PRINCIPAL ---
    def solve(self, max_generations, method_sel='tournament', method_cross='uniform', method_mut='smart', verbose=True):
        self.evaluate() # Evaluar inicial
        
        history = []
        best_global_fit = float('inf')
        
        for gen in range(max_generations):
            
            # 1. IDENTIFICAR ELITES (Antes de que la población cambie)
            # Ordenamos índices por fitness (menor a mayor)
            sorted_indices = np.argsort(self.fitness_scores)
            
            # Guardamos los k mejores (sus genes y su fitness)
            elite_indices = sorted_indices[:self.elitism_count]
            elites_genes = self.population[elite_indices].copy()
            elites_fitness = self.fitness_scores[elite_indices].copy()
            
            # Guardamos historial del mejor absoluto
            current_best_fit = elites_fitness[0]
            history.append(current_best_fit)
            
            if current_best_fit < best_global_fit:
                best_global_fit = current_best_fit
                best_global_sol = elites_genes[0].copy() # El mejor de los elites
                if verbose and gen % 50 == 0:
                    print(f"Gen {gen}: Fitness {int(best_global_fit)}")
                if best_global_fit == 0:
                    if verbose: print(f"✅ ¡SOLUCIÓN EN GEN {gen}!")
                    break

            # 2. EVOLUCIÓN (Crear hijos)
            if method_sel == 'roulette': parents = self.select_roulette()
            else: parents = self.select_tournament()
                
            if method_cross == 'one_point': offspring = self.crossover_one_point(parents)
            else: offspring = self.crossover_uniform(parents)
                
            if method_mut == 'random': offspring = self.mutate_random(offspring)
            else: offspring = self.mutate_smart(offspring)
            
            # 3. REEMPLAZO Y EVALUACIÓN
            self.population = offspring
            self.evaluate() # Evaluamos los nuevos hijos
            
            # 4. APLICAR ELITISMO (Sobrescribir los peores hijos con los elites guardados)
            # Ordenamos la NUEVA población para encontrar a los peores
            sorted_indices_new = np.argsort(self.fitness_scores)
            
            # Los peores están al final de la lista ordenada (índices altos)
            worst_indices = sorted_indices_new[-self.elitism_count:]
            
            # Reemplazamos
            self.population[worst_indices] = elites_genes
            self.fitness_scores[worst_indices] = elites_fitness
            
        return best_global_sol, best_global_fit, history