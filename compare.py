
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les deux fichiers
temps_sat = np.load("temps_sat_glucose.npy")
etapes_act = np.load("checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/evaluator_SUDOKU_step_130200/act_steps.npy")

print(f"Taille originale -> SAT : {len(temps_sat)} | ACT : {len(etapes_act)}")

# 2. Prendre la taille du plus petit tableau pour aligner les deux
taille_min = min(len(temps_sat), len(etapes_act))

temps_sat = temps_sat[:taille_min]
etapes_act = etapes_act[:taille_min]

print(f"Taille corrigée  -> SAT : {len(temps_sat)} | ACT : {len(etapes_act)}")

# 3. Créer le Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(temps_sat, etapes_act, alpha=0.5, color='blue')

plt.title("Corrélation: Temps du Solveur SAT vs Nombre d'étapes ACT (TRM)")
plt.xlabel("Temps (secondes) pris par Glucose3")
plt.ylabel("Nombre d'itérations récursives (ACT)")
plt.grid(True)

plt.savefig("correlation_sat_act.png")
print("Graphique généré : correlation_sat_act.png")