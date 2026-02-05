import numpy as np

from .environment import Environment
from .pso_restart import PSORestart

class PSOSimulatedAnnealing(PSORestart):
    def __init__(self, env:Environment, T0=1000.0, beta=0.95, **kwargs):
        """ kwargs can include num_particles, num_waypoints, max_iter, w, c1, c2, elite_ratio, restart_frequency """
        super().__init__(env, **kwargs)
        self.T = T0
        self.beta = beta
        
    def solve(self, restart=True, **kwargs):
        # kwargs can be used to pass additional parameters to _calculate_fitness
        for k in range(self.max_iter):
            # Evaluation
            for i in range(self.S):
                score = self._calculate_fitness(self.X[i])
                
                # Mise à jour P_best (Standard - on garde toujours le meilleur perso)
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                
                # Mise à jour G_best (Recuit Simulé)
                self._run_simulated_annealing(i, score)
            
            self.history.append(self.G_best_score)
            
            # Refroidissement
            self.T *= self.beta
            
            # 3. Mise à jour Cinématique (Standard)
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
            final_path = np.vstack([self.env.start, self.G_best, self.env.goal])
            return final_path, self.G_best_score, self.history
        else:
            return None, float('inf'), self.history
        
    def _run_simulated_annealing(self, i, score, zero_threshold=1e-9):
        """
        Met à jour le G_best en utilisant le recuit simulé.
        Args:
            i: Index de la particule courante
            score: Score (fitness) de la particule courante
            zero_threshold: Seuil pour éviter les overflow dans le calcul exponentiel
        """
        
        delta = score - self.G_best_score
     
        if delta < 0:
            # Amélioration : On accepte toujours
            self.G_best_score = score
            self.G_best = self.X[i].copy()
        else:
            # Attention aux overflows si T est très petit
            if self.T > zero_threshold:
                prob = np.exp(-delta / self.T)
                if np.random.rand() < prob:
                    # Acceptation de la dégradation
                    self.G_best_score = score
                    self.G_best = self.X[i].copy()