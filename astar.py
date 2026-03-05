import heapq

def manhattan(a, b):
    """ Calcule la distance de Manhattan entre deux points a et b. """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, mode="A*", w=1):
    """ 
    Implémentation flexible : UCS (w=0, g), Greedy (g=0, h), A* (g+h), Weighted A* (g+w*h)
    Retourne : (chemin, nombre_de_noeuds_explores)
    """
    open_list = []
    # (f_score, g_score, position)
    heapq.heappush(open_list, (0, 0, start))
    
    came_from = {start: None}
    g_score = {start: 0}
    nodes_explored = 0 # Compteur de nœuds pour le tableau

    while open_list:
        f, current_g, current = heapq.heappop(open_list)
        nodes_explored += 1

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1], nodes_explored, len(open_list)

        for neighbor in grid.get_neighbors(current):
            tentative_g = current_g + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                h = manhattan(neighbor, goal)
                
                # Sélection de la priorité selon l'expérience demandée
                if mode == "UCS":
                    f_score = tentative_g # f(n) = g(n)
                elif mode == "Greedy":
                    f_score = h # f(n) = h(n)
                else: # A* et Weighted A*
                    f_score = tentative_g + w * h # f(n) = g(n) + w*h(n)
                
                heapq.heappush(open_list, (f_score, tentative_g, neighbor))
                
    return None, nodes_explored, len(open_list)