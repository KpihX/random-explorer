"""PSO with Dimensional Learning improvement.

This module extends PSO with dimensional learning strategy based on:
Xu et al. "Particle swarm optimization based on dimensional learning strategy"
Swarm and Evolutionary Computation, 2019.

When a particle stagnates (no improvement for several iterations),
its personal best is updated dimension by dimension using information
from the global best.
"""

from typing import Any
import numpy as np

from tqdm import tqdm

from ..environment import Environment
from .simulated_annealing import PSOSimulatedAnnealing, SolveResult


class PSODimensionalLearning(PSOSimulatedAnnealing):
    """PSO with dimensional learning for stagnating particles.
    
    If a particle hasn't improved its personal best for a certain
    number of iterations, we try to improve it by copying dimensions
    from the global best one at a time, keeping only beneficial changes.
    
    This reduces oscillation between local and global bests and
    accelerates convergence.
    
    Attributes:
        wait_limit: Iterations without improvement before applying DL.
        stagnation_counter: Per-particle count of stagnant iterations.
    """
    
    def __init__(
        self,
        env: Environment,
        wait_limit: int = 10,
        **kwargs: Any
    ) -> None:
        """Initialize PSO with dimensional learning parameters.
        
        Args:
            env: Environment to plan in.
            wait_limit: Iterations without improvement before DL activation.
            **kwargs: Additional args for PSOSimulatedAnnealing (T0, beta,
                restart_frequency, elite_ratio, num_particles, etc.).
        """
        super().__init__(env, **kwargs)
        self.wait_limit = wait_limit
        # Per-particle stagnation counter
        self.stagnation_counter = np.zeros(self.S)
    
    def solve(
        self,
        simulated_annealing: bool = True,
        restart: bool = True,
        show_progress: bool = True,
        **kwargs: Any
    ) -> SolveResult:
        """Run PSO with dimensional learning.
        
        Args:
            simulated_annealing: Whether to use SA for global best update.
            restart: Whether to enable random restarts.
            show_progress: Whether to show tqdm progress bar.
            **kwargs: Arguments passed to fitness calculation.
            
        Returns:
            Tuple of (final_path, path_length, best_score, history).
        """
        self.T = self.T0
        self.stagnation_counter = np.zeros(self.S)
        
        iterator = tqdm(range(self.max_iter), desc="PSO-DL", disable=not show_progress)
        
        for k in iterator:
            # Evaluate all particles
            for i in range(self.S):
                score, length = self._calculate_fitness(self.X[i], **kwargs)
                
                # Update personal best
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                    self.stagnation_counter[i] = 0  # Reset counter
                else:
                    self.stagnation_counter[i] += 1
                
                # Update global best
                if simulated_annealing:
                    self._update_global_best_sa(i, score, length)
                elif score < self.G_best_score:
                    self.G_best_score = score
                    self.G_best_length = length
                    self.G_best = self.X[i].copy()
            
            # Apply dimensional learning to stagnating particles
            self._apply_dimensional_learning_to_all(**kwargs)
            
            self.history.append(self.G_best_score)
            
            # Cool down temperature (if SA is enabled)
            if simulated_annealing:
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
    
    def _apply_dimensional_learning_to_all(self, **kwargs: Any) -> None:
        """Apply dimensional learning to all stagnating particles.
        
        Args:
            **kwargs: Arguments passed to fitness calculation.
        """
        if self.G_best is None:
            return
        
        for i in range(self.S):
            if self.stagnation_counter[i] > self.wait_limit:
                self._apply_dimensional_learning(i, **kwargs)
                self.stagnation_counter[i] = 0
    
    def _apply_dimensional_learning(
        self,
        particle_idx: int,
        **kwargs: Any
    ) -> None:
        """Try to improve a particle's P_best using G_best dimensions.
        
        Iterates through each dimension (waypoint coordinate) and tries
        replacing it with the corresponding value from G_best. If the
        replacement improves fitness, it's kept; otherwise reverted.
        
        Args:
            particle_idx: Index of the particle to improve.
            **kwargs: Arguments passed to fitness calculation.
        """
        current_pbest = self.P_best[particle_idx]
        current_score = self.P_best_scores[particle_idx]
        
        # Iterate over all dimensions (waypoints x coordinates)
        for n in range(self.N):  # For each waypoint
            for dim in range(2):  # For x and y
                # Save old value
                old_val = current_pbest[n, dim]
                
                # Try value from G_best
                current_pbest[n, dim] = self.G_best[n, dim]
                
                # Evaluate new fitness
                new_score, _ = self._calculate_fitness(current_pbest, **kwargs)
                
                if new_score < current_score:
                    # Keep the change
                    current_score = new_score
                else:
                    # Revert
                    current_pbest[n, dim] = old_val
        
        # Update score (P_best array was modified in place)
        self.P_best_scores[particle_idx] = current_score
