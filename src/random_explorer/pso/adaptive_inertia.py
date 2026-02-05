"""PSO with Adaptive Inertia Weight improvement.

This module extends PSO with linearly decreasing inertia weight,
which promotes exploration at the beginning and exploitation at the end.
"""

from typing import Any
import numpy as np

from tqdm import tqdm

from ..environment import Environment
from .dimensional_learning import PSODimensionalLearning, SolveResult


class PSOAdaptiveInertia(PSODimensionalLearning):
    """PSO with linearly decreasing inertia weight.
    
    The inertia weight decreases linearly from w_max to w_min over
    the course of iterations:
    
        w(k) = w_max - (w_max - w_min) * (k / max_iter)
    
    This promotes exploration at the beginning (high inertia) and
    exploitation/refinement at the end (low inertia).
    
    Attributes:
        w_max: Maximum inertia weight (start of optimization).
        w_min: Minimum inertia weight (end of optimization).
    """
    
    def __init__(
        self,
        env: Environment,
        w_max: float = 0.9,
        w_min: float = 0.4,
        **kwargs: Any
    ) -> None:
        """Initialize PSO with adaptive inertia parameters.
        
        Args:
            env: Environment to plan in.
            w_max: Initial (maximum) inertia weight.
            w_min: Final (minimum) inertia weight.
            **kwargs: Additional args for PSODimensionalLearning (wait_limit,
                T0, beta, restart_frequency, elite_ratio, num_particles, etc.).
        """
        # Initialize with w_max as starting inertia
        super().__init__(env, **kwargs)
        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max  # Override any w passed in kwargs
    
    def solve(
        self,
        simulated_annealing: bool = True,
        restart: bool = True,
        dimensional_learning: bool = True,
        show_progress: bool = True,
        **kwargs: Any
    ) -> SolveResult:
        """Run PSO with adaptive inertia weight.
        
        The inertia weight decreases linearly from w_max to w_min
        over the iterations. Early stopping is applied when a valid
        path is found (score == path_length, i.e., no collision penalty).
        
        Args:
            simulated_annealing: Whether to use SA for global best update.
            restart: Whether to enable random restarts.
            dimensional_learning: Whether to apply dimensional learning.
            show_progress: Whether to show tqdm progress bar.
            **kwargs: Arguments passed to fitness calculation.
            
        Returns:
            Tuple of (final_path, path_length, best_score, history).
        """
        self.T = self.T0
        self.stagnation_counter = np.zeros(self.S)
        
        iterator = tqdm(range(self.max_iter), desc="PSO-Adaptive", disable=not show_progress)
        
        for k in iterator:
            # Update inertia weight linearly
            self.w = self.w_max - (self.w_max - self.w_min) * (k / self.max_iter)
            
            # Evaluate all particles
            for i in range(self.S):
                score, length = self._calculate_fitness(self.X[i], **kwargs)
                
                # Update personal best
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                    self.stagnation_counter[i] = 0
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
            if dimensional_learning:
                self._apply_dimensional_learning_to_all(**kwargs)
            
            self.history.append(self.G_best_score)
            
            # Early stopping: valid path found (score == length, no collision)
            if self.G_best_length is not None and np.isclose(self.G_best_score, self.G_best_length):
                break
            
            # Cool down temperature (if SA is enabled)
            if simulated_annealing:
                self.T *= self.beta
            
            # Update velocities and positions with current w
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
