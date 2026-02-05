import numpy as np

from .environment import Environment
from .pso_simulated_annealing import PSOSimulatedAnnealing

class PSO_DimensionalLearning(PSOSimulatedAnnealing):
    def __init__(self, env:Environment, wait_limit=10, **kwargs):
        """ kwargs can include num_particles, num_waypoints, max_iter, w, c1, c2, elite_ratio, restart_frequency, T0, beta """
        super().__init__(env, **kwargs)
        self.wait_limit = wait_limit
        # Compteur de stagnation pour chaque particule
        self.stagnation_counter = np.zeros(self.num_particles)

    def solve(self, simulated_annealing=True, restart=True, **kwargs):
        for k in range(self.max_iter):
            # Évaluation
            for i in range(self.S):
                score = self._calculate_fitness(self.X[i])
                
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                    self.stagnation_counter[i] = 0 # Reset compteur
                else:
                    self.stagnation_counter[i] += 1

                if simulated_annealing:
                    # Mise à jour G_best (Recuit Simulé)
                    self._run_simulated_annealing(i, score)
                else:
                    # Mise à jour G_best standard
                    if score < self.G_best_score:
                        self.G_best_score = score
                        self.G_best = self.X[i].copy()
            
            # Dimensional Learning sur les particules qui stagnent
            self._run_dimensional_learning()
            
            self.history.append(self.G_best_score)
            
            # Mouvement (Standard)
            r1 = np.random.rand(self.S, self.N, 2)
            r2 = np.random.rand(self.S, self.N, 2)
            self.V = (self.w * self.V) + (self.c1 * r1 * (self.P_best - self.X)) + (self.c2 * r2 * (self.G_best - self.X))
            self.X = self.X + self.V

            self.X[:, :, 0] = np.clip(self.X[:, :, 0], 0, self.env.x_max)
            self.X[:, :, 1] = np.clip(self.X[:, :, 1], 0, self.env.y_max)

            # Random restart if activated
            if restart:
                self._run_random_restart(k)
        
        if self.G_best is not None:
            return np.vstack([self.env.start, self.G_best, self.env.goal]), self.G_best_score, self.history
        return None, float('inf'), self.history

    def _run_dimensional_learning(self):
        """Applique l'apprentissage dimensionnel à toutes les particules."""
        if self.G_best is None:
            return
        
        for i in range(self.S):
            if self.stagnation_counter[i] > self.wait_limit:
                self._apply_dimensional_learning(i)
                self.stagnation_counter[i] = 0

    def _apply_dimensional_learning(self, idx):
        """Essaie d'améliorer P_best[idx] en copiant des dimensions de G_best."""
        # P_best est de forme (N, 2). On itère sur les waypoints et les coords
        current_pbest = self.P_best[idx] # .copy()
        current_score = self.P_best_scores[idx]
        
        for n in range(self.N): # Pour chaque Waypoint
            for dim in range(2): # Pour x et y
                # Sauvegarde ancienne valeur
                old_val = current_pbest[n, dim]
                # Injection valeur de G_best
                current_pbest[n, dim] = self.G_best[n, dim]
                
                # Test
                new_score = self._calculate_fitness(current_pbest)
                
                if new_score < current_score:
                    # On garde le changement (Apprentissage réussi)
                    current_score = new_score
                    # On ne remet pas old_val
                else:
                    # On annule (Revert)
                    current_pbest[n, dim] = old_val
        
        # Mise à jour finale
        # self.P_best[idx] = current_pbest
        self.P_best_scores[idx] = current_score