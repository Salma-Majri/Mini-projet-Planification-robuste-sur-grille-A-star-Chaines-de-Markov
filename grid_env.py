import matplotlib.pyplot as plt
import numpy as np

class GridWorld:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start = None
        self.goal = None

    def is_valid(self, pos):
        """ Vérifie si la position est dans la grille et n'est pas un obstacle """
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos not in self.obstacles

    def get_neighbors(self, pos):
        """ Retourne les voisins accessibles (4-voisins) """
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def plot_grid(self, path=None, title="Grille de Navigation"):
        """ Affiche la grille, les obstacles et le chemin trouvé """
        grid_vis = np.zeros((self.height, self.width))
        for obs in self.obstacles:
            grid_vis[obs[1], obs[0]] = 1 # Obstacles en noir
            
        plt.figure(figsize=(8, 6))
        plt.imshow(grid_vis, cmap='Greys', origin='lower')
        
        # Affichage du départ et de l'arrivée
        if self.start: plt.plot(self.start[0], self.start[1], 'go', markersize=15, label="Départ (s0)")
        if self.goal: plt.plot(self.goal[0], self.goal[1], 'ro', markersize=15, label="But (Goal)")
        
        # Affichage du chemin si présent
        if path:
            px, py = zip(*path)
            plt.plot(px, py, 'b-', linewidth=3, label="Chemin A*")
            
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()