from src.solver import GraphColoringSolver

def find_optimal_coloring(graph, start_colors, pop_size=200, max_generations=1000, mutation_rate=0.05, elitism_count=2, selection='tournament', crossover='uniform', mutation='smart'):
    current_k = start_colors
    best_solution_found = None
    best_k_found = None
    best_history = []
    
    print(f"--- INICIANDO (Start k={start_colors}, Sel={selection}, Cross={crossover}) ---")
    
    while True:
        solver = GraphColoringSolver(
            graph, 
            n_colors=current_k, 
            pop_size=pop_size, 
            mutation_rate=mutation_rate,
            elitism_count=elitism_count
        )
        
        solution, conflicts, history = solver.solve(
            max_generations=max_generations, 
            method_sel=selection, 
            method_cross=crossover, 
            method_mut=mutation,
            verbose=False
        )
        
        if conflicts == 0:
            best_solution_found = solution
            best_k_found = current_k
            best_history = history
            current_k -= 1 
        else:
            break
    return best_solution_found, best_k_found, best_history