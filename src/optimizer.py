import numpy as np
from src.solver import GraphColoringSolver

def find_optimal_coloring(graph, start_colors, pop_size=200, max_generations=1000, mutation_rate=0.05, elitism_count=2):
    """
    Ahora acepta elitism_count.
    """
    current_k = start_colors
    best_solution_found = None
    best_k_found = None
    
    print(f"--- INICIANDO (Start k={start_colors}, Pop={pop_size}, Mut={mutation_rate}, Elites={elitism_count}) ---")
    
    while True:
        print(f"\n>> Probando con {current_k} colores...")
        
        # Pasamos el parámetro de elitismo
        solver = GraphColoringSolver(
            graph, 
            n_colors=current_k, 
            pop_size=pop_size, 
            mutation_rate=mutation_rate,
            elitism_count=elitism_count
        )
        
        solution, conflicts, _ = solver.solve(max_generations, verbose=False)
        
        if conflicts == 0:
            print(f"✅ ÉXITO con {current_k} colores.")
            best_solution_found = solution
            best_k_found = current_k
            current_k -= 1 
        else:
            print(f"❌ FALLO con {current_k} colores.")
            break
            
    return best_solution_found, best_k_found