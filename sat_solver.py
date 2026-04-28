import numpy as np
import time
from pysat.solvers import Glucose3
from tqdm import tqdm # pour avoir une barre de progression
from concurrent.futures import ProcessPoolExecutor, as_completed
def solve_sudoku_sat(grid):
    """
    Résout une grille 9x9 avec le solveur SAT Glucose3 et retourne le temps pris.
    grid: matrice 9x9 où 0 = case vide, et 1-9 = chiffres.
    """
    solver = Glucose3()
    
    # 1. Chaque case doit avoir au moins un chiffre (1 à 9)
    for r in range(9):
        for c in range(9):
            solver.add_clause([int(f"{r+1}{c+1}{v}") for v in range(1, 10)])

    # 2. Contraintes d'unicité (Logique classique SAT pour le Sudoku)
    # Lignes et Colonnes
    for i in range(9):
        for v in range(1, 10):
            for j in range(8):
                for k in range(j + 1, 9):
                    # Pas deux fois la même valeur sur la ligne i
                    solver.add_clause([-int(f"{i+1}{j+1}{v}"), -int(f"{i+1}{k+1}{v}")])
                    # Pas deux fois la même valeur sur la colonne i
                    solver.add_clause([-int(f"{j+1}{i+1}{v}"), -int(f"{k+1}{i+1}{v}")])

    # Blocs 3x3
    for br in range(3):
        for bc in range(3):
            for v in range(1, 10):
                cells = []
                for i in range(3):
                    for j in range(3):
                        cells.append((br * 3 + i + 1, bc * 3 + j + 1))
                for idx1 in range(len(cells)):
                    for idx2 in range(idx1 + 1, len(cells)):
                        r1, c1 = cells[idx1]
                        r2, c2 = cells[idx2]
                        solver.add_clause([-int(f"{r1}{c1}{v}"), -int(f"{r2}{c2}{v}")])

    # 3. Ajouter les indices initiaux de la grille à résoudre
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                solver.add_clause([int(f"{r+1}{c+1}{grid[r][c]}")])
    
    # Résolution et chronométrage
    start_time = time.perf_counter()
    solved = solver.solve()
    end_time = time.perf_counter()
    
    solver.delete() # Libérer la mémoire
    
    if not solved:
        print("Attention, grille insoluble trouvée !")
        
    return end_time - start_time

# --- SCRIPT PRINCIPAL ---

if __name__ == '__main__':
    # 1. Charger et préparer les données
    inputs_bruts = np.load("data/sudoku-extreme-1k-aug-1000/test/all__inputs.npy")
    inputs_corriges = inputs_bruts - 1
    inputs_9x9 = inputs_corriges.reshape(-1, 9, 9)

    temps_resolution = []
    
    print(f"Lancement sur {len(inputs_9x9)} grilles avec tous les cœurs du CPU...")

    # 2. Utiliser ProcessPoolExecutor pour distribuer le travail sur tous les cœurs
    with ProcessPoolExecutor() as executor:
        # On lance toutes les tâches en parallèle
        futures = {executor.submit(solve_sudoku_sat, grille): grille for grille in inputs_9x9}
        
        # On récupère les résultats au fur et à mesure qu'ils se terminent (avec barre de progression)
        for future in tqdm(as_completed(futures), total=len(inputs_9x9)):
            temps = future.result()
            temps_resolution.append(temps)

    # 3. Sauvegarde
    temps_resolution = np.array(temps_resolution)
    np.save("temps_sat_glucose.npy", temps_resolution)
    
    print("\n=== RÉSULTATS SAT ===")
    print(f"Temps moyen  : {np.mean(temps_resolution):.5f} secondes")
    print(f"Temps max    : {np.max(temps_resolution):.5f} secondes")