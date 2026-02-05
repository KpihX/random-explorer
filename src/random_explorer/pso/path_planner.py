"""Particle Swarm Optimization (PSO) path planner.

This module implements the basic PSO algorithm for path planning problems.
It supports both hard and soft penalty modes for collision handling.
"""

from typing import Tuple, List, Optional, Any
import numpy as np

from tqdm import tqdm

from ..environment import Environment


# Type aliases
PathArray = np.ndarray
History = List[float]
FitnessResult = Tuple[float, float]  # (score, path_length)
SolveResult = Tuple[Optional[PathArray], float, float, History]  # (path, length, score, history)


class PSOPathPlanner:
    """PSO-based path planner for 2D environments.
    
    This class implements Particle Swarm Optimization to find optimal paths
    between start and goal positions while avoiding obstacles.
    
    The algorithm supports two penalty modes:
    - Hard mode: Binary penalty (collision or not)
    - Soft mode: Penalty proportional to collision depth (preferred)
    
    Attributes:
        env: The environment to plan in.
        S: Number of particles (swarm size).
        N: Number of waypoints per path.
        max_iter: Maximum number of iterations.
        w: Inertia weight.
        c1: Cognitive coefficient (attraction to personal best).
        c2: Social coefficient (attraction to global best).
        X: Current positions of all particles, shape (S, N, 2).
        V: Current velocities of all particles, shape (S, N, 2).
        P_best: Personal best positions, shape (S, N, 2).
        P_best_scores: Personal best scores, shape (S,).
        G_best: Global best position, shape (N, 2).
        G_best_score: Global best score.
        history: History of global best scores per iteration.
    """
    
    # Default penalty constants
    BASE_PENALTY = 1000.0           # Fixed penalty if any collision exists
    COLLISION_PENALTY_HARD = 100.0  # Penalty per collision (hard mode)
    COLLISION_PENALTY_SOFT = 500.0  # Penalty per unit length in obstacle (soft mode)
    
    def __init__(
        self,
        env: Environment,
        num_particles: int = 50,
        num_waypoints: int = 5,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4
    ) -> None:
        """Initialize the PSO path planner.
        
        Args:
            env: Environment to plan in.
            num_particles: Number of particles in the swarm.
            num_waypoints: Number of intermediate waypoints per path.
            max_iter: Maximum number of iterations.
            w: Inertia weight (controls velocity persistence).
            c1: Cognitive coefficient (personal best attraction).
            c2: Social coefficient (global best attraction).
        """
        self.env = env
        self.S = num_particles
        self.N = num_waypoints
        self.max_iter = max_iter
        
        # PSO hyperparameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize particle positions randomly in environment
        self.X = np.random.rand(self.S, self.N, 2)
        self.X[:, :, 0] *= self.env.x_max
        self.X[:, :, 1] *= self.env.y_max
        
        # Initialize velocities to zero
        self.V = np.zeros((self.S, self.N, 2))
        
        # Initialize personal bests
        self.P_best = self.X.copy()
        self.P_best_scores = np.full(self.S, np.inf)
        
        # Initialize global best
        self.G_best: Optional[np.ndarray] = None
        self.G_best_score = np.inf
        self.G_best_length = np.inf  # Path length of G_best (without penalties)
        
        # History for convergence tracking
        self.history: History = []
        
        # Final results (set after solve())
        self.final_path: Optional[np.ndarray] = None
        self.path_length: float = np.inf
    
    def _calculate_fitness(
        self,
        particle: np.ndarray,
        soft_mode: bool = True,
        base_penalty: Optional[float] = None,
        collision_penalty: Optional[float] = None
    ) -> FitnessResult:
        """Calculate fitness (cost) of a particle's path.
        
        Unified fitness calculation for both modes:
        - Soft mode: penalty based on collision length
        - Hard mode: penalty based on collision count
        
        Formula: path_length + base_penalty * has_collision + collision_metric * collision_penalty
        
        Args:
            particle: Waypoints array of shape (N, 2).
            soft_mode: If True, use collision length; otherwise collision count.
            base_penalty: Fixed penalty if any collision exists.
            collision_penalty: Penalty per unit (length for soft, count for hard).
            
        Returns:
            Tuple of (fitness_score, path_length) where:
                - fitness_score: Fitness value including penalties (lower is better).
                - path_length: Actual geometric path length (without penalties).
        """
        # Set default penalties based on mode
        if base_penalty is None:
            base_penalty = self.BASE_PENALTY
        if collision_penalty is None:
            collision_penalty = self.COLLISION_PENALTY_SOFT if soft_mode else self.COLLISION_PENALTY_HARD
        
        # Build full path: start -> waypoints -> goal
        full_path = np.vstack([self.env.start, particle, self.env.goal])
        
        total_length = Environment.compute_path_length(full_path)
        collision_metric = self.env.evaluate_path_collision(full_path, soft_mode)
        
        # Calculate score
        has_collision = collision_metric > 0
        score = (
            total_length +
            base_penalty * has_collision +
            collision_metric * collision_penalty
        )
        
        return score, total_length
    
    def solve(self, show_progress: bool = True, **kwargs: Any) -> SolveResult:
        """Run PSO algorithm to find optimal path.
        
        Args:
            show_progress: Whether to show tqdm progress bar.
            **kwargs: Arguments passed to fitness calculation
                (soft_mode, base_penalty, penalty_weight, etc.)
            
        Returns:
            Tuple of (final_path, path_length, best_score, history) where:
                - final_path: Array of shape (N+2, 2) or None if not found
                - path_length: Geometric length of the path (without penalties)
                - best_score: Best fitness value achieved (including penalties)
                - history: List of best scores per iteration
        """
        iterator = tqdm(range(self.max_iter), desc="PSO", disable=not show_progress)
        
        for _ in iterator:
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
        
        # Store and return best path found
        if self.G_best is not None:
            self.final_path = np.vstack([self.env.start, self.G_best, self.env.goal])
            self.path_length = self.G_best_length
            return self.final_path, self.path_length, self.G_best_score, self.history
        
        self.final_path = None
        self.path_length = np.inf
        return None, np.inf, np.inf, self.history
    
    def is_path_valid(self, tolerance: float = 1e-6) -> bool:
        """Check if the best path found is collision-free.
        
        A path is valid if score == path_length (no penalty was added).
        
        Args:
            tolerance: Floating point comparison tolerance.
            
        Returns:
            True if no collision penalty was applied, False otherwise.
        """
        if self.final_path is None:
            return False
        return abs(self.G_best_score - self.path_length) < tolerance
