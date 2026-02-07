"""Benchmark utilities for path planning algorithms.

This module provides tools to run and compare different path planning
algorithms across multiple scenarios with consistent metrics.
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .environment import Environment
from .pso import (
    PSOPathPlanner,
    PSORestart,
    PSOSimulatedAnnealing,
    PSODimensionalLearning,
    PSOAdaptiveInertia,
)
from .rrt_planner import RRTPlanner
from .utils import Console


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.
    
    Attributes:
        algorithm: Name of the algorithm used.
        scenario: Name/path of the scenario file.
        path_length: Length of the found path (inf if not found).
        is_valid: Whether the path avoids all obstacles.
        iterations: Number of iterations executed.
        cpu_time: CPU time in seconds.
        path: The actual path found (list of points).
        score: The final score (path_length + penalty).
        history: Convergence history (list of best scores per iteration).
    """
    algorithm: str
    scenario: str
    path_length: float
    is_valid: bool
    iterations: int
    cpu_time: float
    path: Optional[np.ndarray] = None
    score: Optional[float] = None
    history: Optional[List[float]] = None

class Benchmark:
    """Benchmark runner for path planning algorithms.
    
    Provides methods to run single or comparative benchmarks across
    multiple algorithms and scenarios.
    
    Attributes:
        console: Console for output display.
    """
    
    # Default PSO parameters
    DEFAULT_PSO_PARAMS = {
        'num_particles': 300,
        'num_waypoints': 8,
        'max_iter': 100,
        'w': 0.8,
        'c1': 1.6,
        'c2': 1.0
    }
    
    # Default RRT parameters
    DEFAULT_RRT_PARAMS = {
        'max_iter': 3000,
        'delta_s': 30.0,
        'delta_r': 60.0,
        'goal_bias': 0.05
    }
    
    def __init__(self) -> None:
        """Initialize benchmark runner."""
        self.console = Console()
    
    def run_pso(
        self,
        env: Environment,
        planner_class: Type[PSOPathPlanner] = PSOPathPlanner,
        params: Optional[Dict[str, Any]] = None,
        solve_kwargs: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Run a PSO-based algorithm on an environment.
        
        Args:
            env: Environment to plan in.
            planner_class: PSO planner class to use.
            params: Constructor parameters (uses defaults if None).
            solve_kwargs: Arguments to pass to solve().
            
        Returns:
            BenchmarkResult with performance metrics.
        """
        # Merge with defaults
        final_params = self.DEFAULT_PSO_PARAMS.copy()
        if params:
            final_params.update(params)
        
        if solve_kwargs is None:
            solve_kwargs = {'soft_mode': True}
        
        # Create planner
        # print(f"Creating planner {planner_class.__name__} with params: {final_params}")
        planner = planner_class(env, **final_params)
        
        # Run and time
        start_time = time.process_time()
        path, path_length, score, history = planner.solve(**solve_kwargs)
        cpu_time = time.process_time() - start_time
        
        return BenchmarkResult(
            algorithm=planner_class.__name__,
            scenario="",  # Set by caller
            path_length=path_length,
            is_valid=planner.is_path_valid(), 
            iterations=len(history),
            cpu_time=cpu_time,
            path=path,
            score=score,
            history=history
        )
    
    def run_rrt(
        self,
        env: Environment,
        optimized: False,
        intelligent: False,
        params: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Run RRT* algorithm on an environment.
        
        Args:
            env: Environment to plan in.
            params: Constructor parameters (uses defaults if None).
            
        Returns:
            BenchmarkResult with performance metrics.
        """
        # Merge with defaults
        final_params = self.DEFAULT_RRT_PARAMS.copy()
        if params:
            final_params.update(params)
        
        # Create planner
        planner = RRTPlanner(env, **final_params)
        
        # Run and time (disable progress bar in benchmark)
        start_time = time.process_time()
        path_list, path_length, iterations = planner.solve(optimized, intelligent, show_progress=True)
        cpu_time = time.process_time() - start_time
        
        # Convert path to array
        path = np.array(path_list) if path_list else None
        
        # Check path validity
        is_valid = path is not None # and env.evaluate_path_collision(path) == 0
        
        return BenchmarkResult(
            algorithm="RRTPlanner",
            scenario="",
            path_length=path_length,
            is_valid=is_valid,
            iterations=int(iterations),
            cpu_time=cpu_time,
            path=path
        )
    
    def run_all_pso_variants(
        self,
        env: Environment,
        base_params: Dict[str, Any],
        variant_params: Optional[Dict[str, Dict[str, Any]]] = None,
        variant_solve_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        soft_mode: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """Run all PSO variants on an environment with shared base params.
        
        Args:
            env: Environment to plan in.
            base_params: Base PSO params (num_particles, num_waypoints, max_iter, w, c1, c2).
            variant_params: Extra params per variant (e.g., restart_frequency for PSORestart).
            variant_solve_kwargs: Extra solve() kwargs per variant.
            soft_mode: Whether to use soft collision mode.
            
        Returns:
            Dict mapping variant name to BenchmarkResult.
        """
        if variant_params is None:
            variant_params = {}
        if variant_solve_kwargs is None:
            variant_solve_kwargs = {}
        
        # Default variant-specific params
        default_variant_params = {
            'PSOPathPlanner': {},
            'PSORestart': {'restart_frequency': 50, 'elite_ratio': 0.4},
            'PSOSimulatedAnnealing': {'T0': 50, 'beta': 0.9, 'restart_frequency': 50, 'elite_ratio': 0.7},
            'PSODimensionalLearning': {'wait_limit': 50, 'T0': 100, 'beta': 0.7, 'restart_frequency': 50, 'elite_ratio': 0.9},
            'PSOAdaptiveInertia': {'w_max': 0.9, 'w_min': 0.4, 'wait_limit': 25, 'T0': 100, 'beta': 0.7, 'restart_frequency': 30, 'elite_ratio': 0.5},
        }
        
        # Default solve kwargs (disable features for fair comparison)
        default_solve_kwargs = {
            'PSOPathPlanner': {},
            'PSORestart': {},
            'PSOSimulatedAnnealing': {'restart': False},
            'PSODimensionalLearning': {'simulated_annealing': False, 'restart': False},
            'PSOAdaptiveInertia': {'simulated_annealing': False, 'restart': False, 'dimensional_learning': False},
        }
        
        variants = [
            ('PSOPathPlanner', PSOPathPlanner),
            ('PSORestart', PSORestart),
            ('PSOSimulatedAnnealing', PSOSimulatedAnnealing),
            ('PSODimensionalLearning', PSODimensionalLearning),
            ('PSOAdaptiveInertia', PSOAdaptiveInertia),
        ]
        
        results = {}
        
        for name, cls in variants:
            # Build params for this variant
            params = base_params.copy()
            
            # For adaptive, remove 'w'
            if name == 'PSOAdaptiveInertia' and 'w' in params:
                del params['w']
            
            # Add default variant params
            params.update(default_variant_params.get(name, {}))
            
            # Override with user-provided variant params
            params.update(variant_params.get(name, {}))
            
            # Build solve kwargs
            solve_kwargs = {'soft_mode': soft_mode}
            solve_kwargs.update(default_solve_kwargs.get(name, {}))
            solve_kwargs.update(variant_solve_kwargs.get(name, {}))
            
            # print(f"Running {name} with params: {params} and solve_kwargs: {solve_kwargs}")
            self.console.display(f"Running {name} with params: {params} and solve_kwargs: {solve_kwargs}", "PSO Benchmark")
            result = self.run_pso(env, planner_class=cls, params=params, solve_kwargs=solve_kwargs)
            result.algorithm = name
            results[name] = result
        
        return results
    
    def display_results(
        self, 
        results: List[BenchmarkResult],
        title: Optional[str] = None,
        sort_by: str = 'length'
    ) -> None:
        """Display benchmark results in a formatted table.
        
        Args:
            results: List of benchmark results to display.
            title: Custom title for the table.
            sort_by: Sort key ('length', 'score', 'time').
        """
        # Check if any result has score (PSO results)
        has_score = any(r.score is not None for r in results)
        
        if has_score:
            header = (
                f"{'Algorithm':<25} {'Score':>10} {'Length':>10} {'Valid':>6} "
                f"{'Iter':>6} {'Time (s)':>10}"
            )
        else:
            header = (
                f"{'Algorithm':<25} {'Length':>10} {'Valid':>6} "
                f"{'Iter':>6} {'Time (s)':>10}"
            )
        separator = "-" * len(header)
        
        # Sort results
        if sort_by == 'length':
            results = sorted(results, key=lambda r: r.path_length)
        elif sort_by == 'score':
            results = sorted(results, key=lambda r: r.score if r.score else float('inf'))
        elif sort_by == 'time':
            results = sorted(results, key=lambda r: r.cpu_time)
        
        lines = [header, separator]
        
        for r in results:
            length_str = f"{r.path_length:.2f}" if r.path_length < float('inf') else "N/A"
            valid_str = "✓" if r.is_valid else "✗"
            
            if has_score:
                score_str = f"{r.score:.2f}" if r.score is not None else "N/A"
                line = (
                    f"{r.algorithm:<25} {score_str:>10} {length_str:>10} {valid_str:>6} "
                    f"{r.iterations:>6} {r.cpu_time:>10.4f}"
                )
            else:
                line = (
                    f"{r.algorithm:<25} {length_str:>10} {valid_str:>6} "
                    f"{r.iterations:>6} {r.cpu_time:>10.4f}"
                )
            lines.append(line)
        
        display_title = title or f"Benchmark Results: {results[0].scenario if results else 'N/A'}"
        self.console.display(
            "\n".join(lines),
            title=display_title,
            border_style="blue"
        )
    
    def plot_path(
        self,
        result: BenchmarkResult,
        env: Environment,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        show_legend: bool = True
    ) -> Axes:
        """Plot a single path result on an environment.
        
        Args:
            result: Benchmark result containing the path.
            env: Environment to plot (obstacles, start, goal).
            ax: Matplotlib axes to plot on. If None, creates new figure.
            title: Custom title. If None, auto-generates from result.
            show_legend: Whether to show legend.
            
        Returns:
            The matplotlib Axes object used for plotting.
        """
        # Determine title
        if title is None:
            valid_str = "✓" if result.is_valid else "✗"
            title = f"{result.scenario} - L={result.path_length:.1f} {valid_str}"
        
        # Determine path color based on validity
        path_color = 'green' if result.is_valid else 'red'
        
        # Use Environment's plot method
        _, ax = env.plot_environment(
            path=result.path,
            title=title,
            ax=ax,
            path_color=path_color,
            path_label=result.algorithm,
            show_legend=show_legend
        )
        
        return ax
    
    def plot_results_grid(
        self,
        results: Dict[str, BenchmarkResult],
        envs: Dict[str, Environment],
        n_cols: int = 2,
        figsize_per_plot: tuple = (7, 5)
    ) -> Figure:
        """Plot multiple results in a grid layout.
        
        Args:
            results: Dictionary mapping scenario names to BenchmarkResult.
            envs: Dictionary mapping scenario names to Environment.
            n_cols: Number of columns in the grid.
            figsize_per_plot: (width, height) per subplot.
            
        Returns:
            The matplotlib Figure object.
        """
        n_scenarios = len(results)
        n_rows = math.ceil(n_scenarios / n_cols)
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
        )
        
        # Handle single row/column case
        if n_scenarios == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            env = envs[name]
            
            valid_str = "✓" if result.is_valid else "✗"
            title = f"{name} - L={result.path_length:.1f} {valid_str}"
            
            self.plot_path(result, env, ax=ax, title=title, show_legend=False)
        
        # Hide unused subplots
        for idx in range(n_scenarios, len(axes)):
            axes[idx].set_visible(False)
        
        # Single legend for the whole figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        return fig


class Performance:
    """Legacy performance class for backward compatibility.
    
    Deprecated: Use Benchmark class instead.
    """
    
    def __init__(
        self,
        S: int,
        N: int,
        w: float,
        c1: float,
        c2: float,
        max_iter: int,
        file_path: str
    ) -> None:
        """Run a single PSO benchmark and display results.
        
        Args:
            S: Number of particles.
            N: Number of waypoints.
            w: Inertia weight.
            c1: Cognitive coefficient.
            c2: Social coefficient.
            max_iter: Maximum iterations.
            file_path: Path to scenario file.
        """
        benchmark = Benchmark()
        env = Environment(file_path)
        
        params = {
            'num_particles': S,
            'num_waypoints': N,
            'max_iter': max_iter,
            'w': w,
            'c1': c1,
            'c2': c2
        }
        
        result = benchmark.run_pso(env, PSOPathPlanner, params)
        result.scenario = file_path
        
        length_str = f"{result.path_length:.2f}" if result.path_length < float('inf') else "N/A"
        benchmark.console.display(
            f"Path length: {length_str}\n"
            f"Iterations: {result.iterations}\n"
            f"CPU time: {result.cpu_time:.4f} s\n"
            f"Valid path: {result.is_valid}",
            title="PSO Performance",
            border_style="green"
        )
