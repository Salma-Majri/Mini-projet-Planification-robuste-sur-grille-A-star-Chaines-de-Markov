import random

def simulate_trajectory(grid, path, epsilon, max_steps=100):
    """ Simule une trajectoire réelle de l'agent sur la grille. """
    # Définition de la politique (actions à prendre sur le chemin)
    policy = {}
    for i in range(len(path) - 1):
        curr, next_node = path[i], path[i+1]
        policy[curr] = (next_node[0]-curr[0], next_node[1]-curr[1])
    
    current_pos = path[0] # Position initiale s0
    steps = 0
    
    # Boucle de simulation jusqu'à atteindre le but ou dépasser le temps limite
    while current_pos != grid.goal and steps < max_steps:
        action = policy.get(current_pos, (0,0))
        if action == (0,0): break # Agent perdu hors trajectoire
        
        dx, dy = action
        r = random.random()
        
        # Sélection stochastique du mouvement selon epsilon
        if r < (1 - epsilon):
            move = (current_pos[0] + dx, current_pos[1] + dy) # Succès
        elif r < (1 - epsilon/2):
            move = (current_pos[0] + dy, current_pos[1] + dx) # Déviation 1
        else:
            move = (current_pos[0] - dy, current_pos[1] - dx) # Déviation 2
        
        # Mise à jour de la position si le mouvement est valide
        if grid.is_valid(move):
            current_pos = move
        # En cas de mur, l'agent reste sur place
            
        steps += 1
    
    # Retourne (Succès?, Nombre de pas)
    return current_pos == grid.goal, steps