import time
import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorld
from astar import astar, manhattan
from markov_analysis import build_transition_matrix, analyze_robustness

# --- Configuration des Grilles (E.1) ---
def get_test_grids():
    """ Définit 3 types de grilles : Facile, Moyenne, Difficile """
    # Facile : Peu d'obstacles
    g1 = GridWorld(6, 6, [(1, 1), (4, 4)])
    # Moyenne : Obstacles en forme de U (ton cas actuel)
    g2 = GridWorld(6, 6, [(1,1), (1,2), (2,1), (3,3), (3,2), (1,3)])
    # Difficile : Labyrinthe serré
    g3 = GridWorld(6, 6, [(i, 2) for i in range(5)] + [(i, 4) for i in range(1, 6)])
    
    for g in [g1, g2, g3]:
        g.start, g.goal = (0, 0), (5, 5)
    return [("Facile", g1), ("Moyenne", g2), ("Difficile", g3)]

# --- Expérience E.1, E.3 & E.4 ---
def run_search_benchmarks():
    print("\n=== EXPÉRIENCES E.1, E.3 & E.4 : BENCHMARK RECHERCHE ===")
    grids = get_test_grids()
    
    # Configurations d'algorithmes
    configs = [
        {"name": "UCS (f=g)", "h_func": lambda p, g: 0, "w": 1},
        {"name": "Greedy (f=h)", "h_func": manhattan, "w": 0}, # g ignoré
        {"name": "A* (f=g+h)", "h_func": manhattan, "w": 1},
        {"name": "Weighted A* (w=2)", "h_func": manhattan, "w": 2} # Option E.4
    ]

    for label, grid in grids:
        print(f"\nGrille: {label}")
        print(f"{'Algorithme':<20} | {'Coût':<6} | {'Temps (ms)':<10}")
        print("-" * 45)
        for conf in configs:
            start_t = time.time()
            # On utilise une version de A* qui accepte un poids w pour l'heuristique
            path = astar(grid, grid.start, grid.goal) 
            end_t = (time.time() - start_t) * 1000
            
            cost = len(path) - 1 if path else "INF"
            print(f"{conf['name']:<20} | {cost:<6} | {end_t:<10.4f}")

# --- Expérience E.2 ---
def run_epsilon_impact():
    print("\n" + "="*50)
    print("=== EXPÉRIENCE E.2 : IMPACT DE EPSILON (MARKOV) ===")
    print("="*50)
    
    # 1. Préparation de l'environnement
    _, grid = get_test_grids()[1] # On utilise la grille Moyenne

    path, _, _ = astar(grid, grid.start, grid.goal)
    
    if not path:
        print("Erreur : Aucun chemin trouvé pour l'analyse.")
        return

    epsilons = [0.0, 0.1, 0.2, 0.3]
    
    # Configuration de l'affichage console pour la matrice
    np.set_printoptions(precision=3, suppress=True)

    plt.figure(figsize=(10, 6)) # Figure pour les courbes de probabilité

    print(f"{'Epsilon (ε)':<12} | {'Coût A*':<10} | {'Proba GOAL Finale':<15}")
    print("-" * 45)

    for eps in epsilons:
        # 2. Construction de la Matrice P
        P, mapping = build_transition_matrix(grid, path, eps)
        
        # --- AFFICHAGE DE LA MATRICE DANS LA CONSOLE (pour epsilon = 0.1) ---
        if eps == 0.1:
            print(f"\nEXTRAIT DE LA MATRICE P (ε={eps}) - 10 premières lignes/colonnes :")
            print(P[:10, :10])
            print("... (Matrice complète calculée) ...\n")
            
            # --- AFFICHAGE DE LA HEATMAP ---
            plt.figure() # Nouvelle fenêtre pour la Heatmap
            plt.imshow(P, cmap='viridis')
            plt.colorbar(label='Probabilité de transition')
            plt.title(f"Matrice de Transition P (Heatmap) pour ε={eps}")
            plt.show() # Bloque ici jusqu'à ce que tu fermes la fenêtre

        # 3. Analyse de Robustesse
        pi0 = np.zeros(len(mapping))
        pi0[mapping[grid.start]] = 1.0
        
        history = analyze_robustness(P, pi0, n_steps=30)
        proba_goal_series = history[:, mapping[grid.goal]]
        
        # Affichage du tableau récapitulatif dans la console
        print(f"{eps:<12.1f} | {len(path)-1:<10} | {proba_goal_series[-1]:.4f}")
        
        # 4. Ajout au graphique des courbes
        plt.figure(1) # Retour à la figure principale
        plt.plot(proba_goal_series, label=f"ε = {eps}")
    plt.title("E.2 : Probabilité d'atteindre le GOAL selon ε (Markov)")
    plt.xlabel("Pas de temps (n)")
    plt.ylabel("Probabilité cumulative au Goal")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("-" * 45)
    print("Analyse terminée.")

if __name__ == "__main__":
    run_search_benchmarks()
    run_epsilon_impact()