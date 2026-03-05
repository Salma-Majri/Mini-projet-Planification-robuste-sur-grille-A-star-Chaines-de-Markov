import time
import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorld
from astar import astar, manhattan
from markov_analysis import build_transition_matrix, analyze_robustness
from simulator import simulate_trajectory
from experiments import get_test_grids

# --- PHASE 1 : Définition de l'environnement (Grille Moyenne) ---
obstacles = [(1,1), (1,2), (2,1), (3,3), (3,2), (1,3)]
grid = GridWorld(6, 6, obstacles)
grid.start, grid.goal = (0,0), (5,5)

# --- PHASE 2 : Planification A* Déterministe ---
print("=== PHASE 2 : RECHERCHE DU CHEMIN OPTIMAL A* ===")
path, nodes, open_nodes = astar(grid, grid.start, grid.goal, mode="A*")
print(f"Chemin trouvé en {nodes} nœuds.")
grid.plot_grid(path, title="Planification Déterministe A*")

# --- PHASE 3 & 4 : Analyse de Markov & Robustesse ---
print("\n=== PHASE 3 & 4 : ANALYSE DE MARKOV (ε=0.15) ===")
epsilon = 0.15 
P, mapping = build_transition_matrix(grid, path, epsilon)

pi0 = np.zeros(len(mapping))
pi0[mapping[grid.start]] = 1.0

history = analyze_robustness(P, pi0, 30)
proba_goal = history[:, mapping[grid.goal]]

# Visualisation Courbe simple (Phase 4)
plt.figure(figsize=(10, 5))
plt.plot(proba_goal, 'b-o', label=f"Probabilité théorique (ε={epsilon})")
plt.xlabel("Pas de temps (n)")
plt.ylabel("Probabilité")
plt.title("Évolution de la probabilité d'atteinte du but (Analyse Markov)")
plt.grid(True)
plt.legend()
plt.show()

# --- ANALYSE DE L'IMPACT DE EPSILON (Expérience E.2) ---
def run_epsilon_impact(grid, path):
    print("\n=== EXPÉRIENCE E.2 : IMPACT DE EPSILON SUR LA ROBUSTESSE ===")
    epsilons = [0.0, 0.1, 0.2, 0.3]
    plt.figure(figsize=(10, 6))
    
    for eps in epsilons:
        P_eps, mapping_eps = build_transition_matrix(grid, path, eps)
        pi0_eps = np.zeros(len(mapping_eps))
        pi0_eps[mapping_eps[grid.start]] = 1.0
        
        history_eps = analyze_robustness(P_eps, pi0_eps, n_steps=30)
        proba_series = history_eps[:, mapping_eps[grid.goal]]
        plt.plot(proba_series, label=f"ε = {eps}")

    plt.title("E.2 : Probabilité d'atteindre le GOAL selon ε (Markov)")
    plt.xlabel("Pas de temps (n)")
    plt.ylabel("Probabilité cumulative au Goal")
    plt.legend()
    plt.grid(True)
    plt.show() 

run_epsilon_impact(grid, path)

# --- PHASE 5 : Simulation Monte-Carlo ---
print("\n=== PHASE 5 : SIMULATION MONTE-CARLO (VALIDATION) ===")
print(f"Lancement de 1000 simulations avec epsilon = {epsilon}...")
n_sim = 1000
success_list = [simulate_trajectory(grid, path, epsilon)[0] for _ in range(n_sim)]
taux_succes = sum(success_list) / n_sim

print(f"Résultat Final :")
print(f"- Probabilité théorique (Markov) : {proba_goal[-1]:.4f}")
print(f"- Taux de succès empirique (Simu) : {taux_succes:.4f}")

# --- EXPÉRIENCES E.1, E.3 & E.4 : BENCHMARK COMPLET ---
def run_search_benchmarks():
    print("\n" + "="*95)
    print("=== EXPÉRIENCES E.1, E.3 & E.4 : BENCHMARK DE PERFORMANCE COMPLET ===")
    print("="*95)
    
    grids = get_test_grids() 
    
    configs = [
        {"name": "UCS (f=g)", "mode": "UCS", "w": 1},
        {"name": "Greedy (f=h)", "mode": "Greedy", "w": 1},
        {"name": "A* (f=g+h)", "mode": "A*", "w": 1},
        {"name": "Weighted A* (w=2)", "mode": "A*", "w": 2}
    ]

    for label, g_test in grids:
        print(f"\nTYPE DE GRILLE : {label}")
        print(f"{'Algorithme':<20} | {'Coût':<6} | {'Nœuds':<6} | {'Open':<6} | {'Succès':<8} | {'Temps (ms)':<10}")
        print("-" * 95)
        
        # Initialisation des données pour le graphique de cette grille
        grid_results = {"names": [], "nodes": [], "costs": []}
        
        for conf in configs:
            start_t = time.time()
            
            # Calcul
            res_path, res_nodes, res_open = astar(g_test, (0,0), (5,5), mode=conf["mode"], w=conf["w"]) 
            
            end_t = (time.time() - start_t) * 1000
            
            # Analyse des résultats
            success = (res_path is not None and len(res_path) > 0 and res_path[-1] == (5,5))
            cost = len(res_path) - 1 if res_path else 0
            
            # Affichage console
            cost_display = cost if res_path else "INF"
            print(f"{conf['name']:<20} | {cost_display:<6} | {res_nodes:<6} | {res_open:<6} | {str(success):<8} | {end_t:<10.4f}")
            
            # Stockage pour le graphique
            grid_results["names"].append(conf["name"])
            grid_results["nodes"].append(res_nodes)
            grid_results["costs"].append(cost)

        # --- GÉNÉRATION DU GRAPHE COMPARATIF (Une fois par type de grille) ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(grid_results["names"]))
        width = 0.35

        # Axe 1 : Nœuds Explorés
        color_nodes = 'skyblue'
        rects1 = ax1.bar(x - width/2, grid_results["nodes"], width, label='Nœuds Explorés', color=color_nodes, edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Algorithmes')
        ax1.set_ylabel('Nombre de Nœuds (Efficacité)', color='steelblue', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(grid_results["names"])
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # Axe 2 : Coût du chemin
        ax2 = ax1.twinx()
        color_costs = 'salmon'
        rects2 = ax2.bar(x + width/2, grid_results["costs"], width, label='Coût du chemin', color=color_costs, edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Coût / Longueur (Optimalité)', color='indianred', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='indianred')

        plt.title(f"Benchmark Performance : Grille {label}", fontsize=14)
        
        # Fusion des légendes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    run_search_benchmarks()