"""PSO with Random Restart improvement.

This module extends the basic PSO algorithm with periodic random restarts
to escape local optima and explore new regions of the search space.
"""

from typing import Any
import numpy as np

from tqdm import tqdm

from ..environment import Environment
from .path_planner import PSOPathPlanner, SolveResult


class PSORestart(PSOPathPlanner):
    """PSO path planner with random restart capability.
    
    Periodically reinitializes most particles to random positions while
    keeping the best (elite) particles. This helps escape local optima.
    
    Attributes:
        restart_frequency: Number of iterations between restarts.
        elite_ratio: Fraction of best particles to preserve (0.0 to 1.0).
        num_restarts: Counter of how many restarts have occurred.
    """
    
    def __init__(
        self,
        env: Environment,
        restart_frequency: int = 20,
        elite_ratio: float = 0.1,
        **kwargs: Any
    ) -> None:
        """Initialize PSO with restart parameters.
        
        Args:
            env: Environment to plan in.
            restart_frequency: Iterations between restarts (0 = disabled).
            elite_ratio: Fraction of particles to preserve during restart.
            **kwargs: Additional args for PSOPathPlanner (num_particles,
                num_waypoints, max_iter, w, c1, c2).
        """
        super().__init__(env, **kwargs)
        self.restart_frequency = restart_frequency
        self.elite_ratio = elite_ratio
        self.num_restarts = 0
    
    def solve(self, show_progress: bool = True, **kwargs: Any) -> SolveResult:
        """Run PSO with periodic random restarts.
        
        Args:
            show_progress: Whether to show tqdm progress bar.
            **kwargs: Arguments passed to fitness calculation.
            
        Returns:
            Tuple of (final_path, path_length, best_score, history).
        """
        iterator = tqdm(range(self.max_iter), desc="PSO-Restart", disable=not show_progress)
        
        for k in iterator:
            # Evaluate all particles
            for i in range(self.S):
                score, length = self._calculate_fitness(self.X[i], **kwargs)
                
                # Update personal best
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                    
                    # Update global best
                    if score < self.G_best_score:
                        self.G_best_score = score
                        self.G_best_length = length
                        self.G_best = self.X[i].copy()
            
            self.history.append(self.G_best_score)
            
            # Update velocities and positions
            r1 = np.random.rand(self.S, self.N, 2)
            r2 = np.random.rand(self.S, self.N, 2)
            
            self.V = (
                self.w * self.V +
                self.c1 * r1 * (self.P_best - self.X) +
                self.c2 * r2 * (self.G_best - self.X)
            )
            self.X = self.X + self.V
            
            # Apply boundary constraints
            self.X[:, :, 0] = np.clip(self.X[:, :, 0], 0, self.env.x_max)
            self.X[:, :, 1] = np.clip(self.X[:, :, 1], 0, self.env.y_max)
            
            # Perform random restart if conditions are met
            self._maybe_restart(k)
        
        # Store and return best path found
        if self.G_best is not None:
            self.final_path = np.vstack([self.env.start, self.G_best, self.env.goal])
            self.path_length = self.G_best_length
            return self.final_path, self.path_length, self.G_best_score, self.history
        
        self.final_path = None
        self.path_length = np.inf
        return None, np.inf, np.inf, self.history
    
    def _maybe_restart(self, iteration: int) -> None:
        """Perform random restart if conditions are met.
        
        Args:
            iteration: Current iteration number (0-indexed).
        """
        if self.restart_frequency <= 0:
            return
        
        # Restart at specified frequency, except on last iteration
        if ((iteration + 1) % self.restart_frequency == 0 and
            iteration < self.max_iter - 1):
            self._random_restart()
            self.num_restarts += 1
    
    def _random_restart(self) -> None:
        """Reinitialize particles while preserving elite ones.
        
        Keeps the best particles (based on P_best scores) and
        reinitializes the rest to random positions.
        """
        # Number of elite particles to preserve
        num_elites = max(1, int(self.S * self.elite_ratio))
        
        # Find elite particles (k smallest scores without full sort)
        elite_indices = np.argpartition(self.P_best_scores, num_elites)[:num_elites]
        
        # Save elite data
        elite_particles = self.X[elite_indices].copy()
        elite_velocities = self.V[elite_indices].copy()
        elite_p_best = self.P_best[elite_indices].copy()
        elite_p_scores = self.P_best_scores[elite_indices].copy()
        
        # Reinitialize all particles
        self.X = np.random.rand(self.S, self.N, 2)
        self.X[:, :, 0] *= self.env.x_max
        self.X[:, :, 1] *= self.env.y_max
        
        # Reset velocities
        self.V = np.zeros((self.S, self.N, 2))
        
        # Reset personal bests
        self.P_best = self.X.copy()
        self.P_best_scores = np.full(self.S, np.inf)
        
        # Restore elite particles
        self.X[:num_elites] = elite_particles
        self.V[:num_elites] = elite_velocities
        self.P_best[:num_elites] = elite_p_best
        self.P_best_scores[:num_elites] = elite_p_scores
