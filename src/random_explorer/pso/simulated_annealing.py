"""PSO with Simulated Annealing improvement.

This module extends PSO with simulated annealing for global best updates,
allowing the algorithm to escape local optima by occasionally accepting
worse solutions with decreasing probability.
"""

from typing import Any
import numpy as np

from tqdm import tqdm

from ..environment import Environment
from .restart import PSORestart, SolveResult


class PSOSimulatedAnnealing(PSORestart):
    """PSO with simulated annealing for global best selection.
    
    Combines PSO with simulated annealing: when evaluating a particle,
    it may become the new global best even if worse than the current
    one, with probability exp(-delta/T) where delta is the fitness
    difference and T is the temperature.
    
    The temperature decreases over time: T(k) = T0 * beta^k
    
    Attributes:
        T: Current temperature.
        T0: Initial temperature.
        beta: Cooling rate (0 < beta < 1).
    """
    
    def __init__(
        self,
        env: Environment,
        T0: float = 1000.0,
        beta: float = 0.95,
        **kwargs: Any
    ) -> None:
        """Initialize PSO with simulated annealing parameters.
        
        Args:
            env: Environment to plan in.
            T0: Initial temperature (higher = more exploration).
            beta: Cooling rate (closer to 1 = slower cooling).
            **kwargs: Additional args for PSORestart (restart_frequency,
                elite_ratio, num_particles, num_waypoints, etc.).
        """
        super().__init__(env, **kwargs)
        self.T0 = T0
        self.T = T0
        self.beta = beta
    
    def solve(self, restart: bool = True, show_progress: bool = True, **kwargs: Any) -> SolveResult:
        """Run PSO with simulated annealing.
        
        Args:
            restart: Whether to enable random restarts.
            show_progress: Whether to show tqdm progress bar.
            **kwargs: Arguments passed to fitness calculation.
            
        Returns:
            Tuple of (final_path, path_length, best_score, history).
        """
        self.T = self.T0
        
        iterator = tqdm(range(self.max_iter), desc="PSO-SA", disable=not show_progress)
        
        for k in iterator:
            # Evaluate all particles
            for i in range(self.S):
                score, length = self._calculate_fitness(self.X[i], **kwargs)
                
                # Update personal best (always keep best)
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                
                # Update global best with simulated annealing
                self._update_global_best_sa(i, score, length)
            
            self.history.append(self.G_best_score)
            
            # Cool down temperature
            self.T *= self.beta
            
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
            
            # Random restart if enabled
            if restart:
                self._maybe_restart(k)
        
        # Store and return best path found
        if self.G_best is not None:
            self.final_path = np.vstack([self.env.start, self.G_best, self.env.goal])
            self.path_length = self.G_best_length
            return self.final_path, self.path_length, self.G_best_score, self.history
        
        self.final_path = None
        self.path_length = float('inf')
        return None, float('inf'), float('inf'), self.history
    
    def _update_global_best_sa(
        self,
        particle_idx: int,
        score: float,
        length: float,
        zero_threshold: float = 1e-9
    ) -> None:
        """Update global best using simulated annealing acceptance.
        
        If the new score is better, always accept. If worse, accept
        with probability exp(-delta/T).
        
        Args:
            particle_idx: Index of the particle being evaluated.
            score: Fitness score of the particle.
            length: Path length of the particle (without penalties).
            zero_threshold: Minimum temperature to avoid overflow.
        """
        delta = score - self.G_best_score
        
        if delta < 0:
            # Improvement: always accept
            self.G_best_score = score
            self.G_best_length = length
            self.G_best = self.X[particle_idx].copy()
        elif self.T > zero_threshold:
            # Worse solution: accept with probability exp(-delta/T)
            prob = np.exp(-delta / self.T)
            if np.random.rand() < prob:
                self.G_best_score = score
                self.G_best_length = length
                self.G_best = self.X[particle_idx].copy()
