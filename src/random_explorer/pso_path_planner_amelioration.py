import numpy as np
import time

from .environment import Environment
from .pso_path_planner import PSOPathPlanner

class PSOPathPlannerUpgrade (PSOPathPlanner):
    COLISION_PENALTY_HARD = 10000.0
    COLISION_WEIGHT_SOFT = 100.0
    BASE_PENALTY_SOFT = 1000.0
    def __init__(self, env:Environment, num_particles=50, num_waypoints=5, max_iter=100, w=0.7, c1=1.4, c2=1.4):
        super().__init__(env, num_particles, num_waypoints, max_iter, w, c1, c2)
        self.num_restarts = 0 # nous permettra de connaitre le nombre de fois qu'on a relancé l'algo

    def solveup(self, restart_frequency=0, elite_ratio=0.1, **kwargs):
        """
        Exécute l'algorithme PSO avec possibilité de random restart.
        
        Args:
            restart_frequency: Fréquence de recommencement 
            elite_ratio: Proportion des bonnes particules à conserver (0.0 à 1.0)
            **kwargs: Arguments passés à _calculate_fitness
        """
        for k in range(self.max_iter):
            for i in range(self.S):
                score = self._calculate_fitness(self.X[i], **kwargs)  
                
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                    
                    if score < self.G_best_score:
                        self.G_best_score = score
                        self.G_best = self.X[i].copy()
            
            self.history.append(self.G_best_score)
            
            # Mise à jour des vitesses et positions
            r1 = np.random.rand(self.S, self.N, 2)
            r2 = np.random.rand(self.S, self.N, 2)
            self.V = (self.w * self.V) + (self.c1 * r1 * (self.P_best - self.X)) + (self.c2 * r2 * (self.G_best - self.X))
            self.X = self.X + self.V
            
            # Conditions aux limites
            self.X[:, :, 0] = np.clip(self.X[:, :, 0], 0, self.env.x_max)
            self.X[:, :, 1] = np.clip(self.X[:, :, 1], 0, self.env.y_max)
            
            # Random restart
            if restart_frequency > 0 and (k + 1) % restart_frequency == 0 and k < self.max_iter - 1:
                self._random_restart(elite_ratio)
                self.num_restarts += 1
        
        if self.G_best is not None:
            final_path = np.vstack([self.env.start, self.G_best, self.env.goal])
            return final_path, self.G_best_score, self.history
        else:
            return None, np.inf, self.history
    
    def _random_restart(self, elite_ratio=0.1):
        """
        Réinitialise aléatoirement la majorité des particules tout en conservant
        les meilleures .
        
        Args:
            elite_ratio: Proportion des meilleures particules à conserver (0.0 à 1.0)
        """
        # Nombre de particules élites à conserver
        num_elites = max(1, int(self.S * elite_ratio))
        
        # Trier les particules par score
        sorted_indices = np.argsort(self.P_best_scores)
        elite_indices = sorted_indices[:num_elites]
        
        # Sauvegarder les particules élites
        elite_particles = self.X[elite_indices].copy()
        elite_velocities = self.V[elite_indices].copy()
        elite_p_best = self.P_best[elite_indices].copy()
        elite_p_scores = self.P_best_scores[elite_indices].copy()
        
        # Réinitialiser TOUTES les particules
        self.X = np.random.rand(self.S, self.N, 2)
        self.X[:, :, 0] *= self.env.x_max
        self.X[:, :, 1] *= self.env.y_max
        
        # Réinitialiser les vitesses
        self.V = np.zeros((self.S, self.N, 2))
        
        # Réinitialiser les meilleures positions personnelles
        self.P_best = self.X.copy()
        self.P_best_scores = np.full(self.S, np.inf)
        
        # Restaurer les particules élites
        self.X[:num_elites] = elite_particles
        self.V[:num_elites] = elite_velocities
        self.P_best[:num_elites] = elite_p_best
        self.P_best_scores[:num_elites] = elite_p_scores

