"""Random Explorer - Path Planning Algorithms.

This package provides implementations of path planning algorithms
including Particle Swarm Optimization (PSO) variants and
Rapidly-exploring Random Trees (RRT*).

Modules:
    environment: Environment parsing and visualization.
    pso: PSO algorithm variants (path_planner, restart, sa, dl, adaptive).
    rrt_planner: RRT* algorithm.
    benchmark: Performance benchmarking utilities.
    utils: Helper utilities.
"""

from .environment import Environment
from .pso import (
    PSOPathPlanner,
    PSORestart,
    PSOSimulatedAnnealing,
    PSODimensionalLearning,
    PSOAdaptiveInertia,
)
from .rrt_planner import RRTPlanner, Node
from .benchmark import Benchmark, BenchmarkResult, Performance
from .utils import Console

__all__ = [
    # Environment
    'Environment',
    # PSO variants
    'PSOPathPlanner',
    'PSORestart',
    'PSOSimulatedAnnealing',
    'PSODimensionalLearning',
    'PSOAdaptiveInertia',
    # RRT
    'RRTPlanner',
    'Node',
    # Benchmark
    'Benchmark',
    'BenchmarkResult',
    'Performance',  # Legacy
    # Utilities
    'Console',
]
