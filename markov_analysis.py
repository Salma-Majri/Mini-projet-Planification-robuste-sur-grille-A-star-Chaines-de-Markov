import numpy as np

def build_transition_matrix(grid, path, epsilon):
    """ Construit la matrice de transition P basée sur la politique induite par le chemin A*. """
    # Liste de tous les états libres (états du système)
    cells = [(x, y) for x in range(grid.width) for y in range(grid.height) if (x,y) not in grid.obstacles]
    cell_to_id = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    
    # Initialisation d'une matrice stochastique par lignes
    P = np.zeros((n, n))

    # Extraction de la politique : quelle action prendre à chaque étape du chemin ?
    policy = {}
    policy = {}
    if path is not None:
        for i in range(len(path) - 1):
            curr = path[i]
            next_node = path[i+1]
            policy[curr] = (next_node[0]-curr[0], next_node[1]-curr[1])

    for cell in cells:
        idx = cell_to_id[cell]
        
        # Le but (Goal) est un état absorbant : p_goal,goal = 1
        if cell == grid.goal:
            P[idx, idx] = 1.0
            continue
            
        action = policy.get(cell, (0,0))
        if action == (0,0): 
            # Si l'agent dévie hors du chemin, il tente de rester sur place ou est bloqué
            P[idx, idx] = 1.0
            continue

        # Modélisation de l'incertitude epsilon
        dx, dy = action
        target = (cell[0] + dx, cell[1] + dy)    # Action voulue (1 - epsilon)
        dev1 = (cell[0] + dy, cell[1] + dx)      # Déviation latérale 1 (epsilon/2)
        dev2 = (cell[0] - dy, cell[1] - dx)      # Déviation latérale 2 (epsilon/2)

        for move, prob in [(target, 1-epsilon), (dev1, epsilon/2), (dev2, epsilon/2)]:
            if grid.is_valid(move):
                P[idx, cell_to_id[move]] += prob
            else:
                # Collision ou obstacle : l'agent reste sur place
                P[idx, idx] += prob
                
    return P, cell_to_id

def analyze_robustness(P, pi0, n_steps):
    """ Calcule l'évolution de la distribution pi(n) = pi(0) * P^n """
    history = []
    current_pi = pi0.copy()
    for _ in range(n_steps):
        current_pi = current_pi @ P 
        history.append(current_pi)
    return np.array(history)

def print_matrix_sample(P, size=10):
    """ Affiche un extrait de la matrice P pour vérification """
    print(f"\nExtrait de la Matrice de Transition P ({size}x{size}) :")
    # On limite l'affichage à un bloc 10x10 pour que ce soit lisible
    sample = P[:size, :size]
    for row in sample:
        print("[" + "  ".join([f"{v:.2f}" if v > 0 else "0.  " for v in row]) + "]")