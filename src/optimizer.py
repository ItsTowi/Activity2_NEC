from src.solver import GraphColoringSolver
MAX_PATIENCE = 2

def find_optimal_coloring(graph, 
                          start_colors,
                          optimal_solution,
                          pop_size=200, 
                          max_generations=1000, 
                          mutation_rate=0.05, 
                          elitism_count=2, 
                          selection='tournament', 
                          crossover='uniform', 
                          mutation='smart'):
    
    best_solution_found = None
    best_k_found = None
    best_history = []
    found = False
    patience = 0
    current_k = start_colors

    # Bucle desde start hasta optima
    while (found == False and current_k >= optimal_solution and patience < MAX_PATIENCE):

        solver = GraphColoringSolver(
            graph, 
            n_colors=current_k, 
            pop_size=pop_size, 
            mutation_rate=mutation_rate,
            elitism_count=elitism_count
        )
        
        # Ejecutamos el solver
        sol_found, fitness, history = solver.solve(
            max_generations=max_generations, 
            method_sel=selection, 
            method_cross=crossover, 
            method_mut=mutation,
            verbose=False
        )
        
        # Verificamos si encontró solución
        # Para eso la fitness tiene que ser como maximo igual a K
        if fitness == optimal_solution:
            best_solution_found = sol_found
            best_k_found = fitness
            best_history = history
            print(f" Solución optima encontrada, colores utilizados: {fitness}")
            found = True
        elif fitness <= current_k:
            print(f" Solución válida encontrada para k={current_k}. (Colores utilizados: {fitness})")
            # Guardamos esta como la mejor hasta el momento
            best_solution_found = sol_found
            best_k_found = fitness
            best_history = history
            if fitness == current_k:
                current_k = current_k - 1
            else:
                current_k = fitness - 1
        else:
            print(f" No se pudo encontrar solución válida para k={current_k} (Fitness: {fitness}).")
            patience = patience + 1
            current_k = current_k - 1
    if (found == False):
        print(f" Solucion final con {best_k_found} colores")
    return best_solution_found, best_k_found, best_history