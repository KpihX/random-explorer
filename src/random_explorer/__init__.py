"""Random Explorer - Path Planning Algorithms.

This package provides implementations of path planning algorithms
including Particle Swarm Optimization (PSO) variants and
Rapidly-exploring Random Trees (RRT*).

Modules:
    environment: Environment parsing and visualization.
    pso: PSO algorithm variants (path_planner, restart, sa, dl, adaptive).
    rrt: RRT* algorithm package.
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
from .rrt import RRTPlanner, Node, MultiRobotRRTPlanner
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
    'MultiRobotRRTPlanner',
    # Benchmark
    'Benchmark',
    'BenchmarkResult',
    'Performance',  # Legacy
    # Utilities
    'Console',
]
