"""Eikonale Explorer - Path Planning via Eikonal Equation.

This package provides implementations of:
- Eikonal equation solver (Lax-Friedrichs scheme)
- Path reconstruction (Euler, Heun gradient descent)
- Various index functions (uniform, diopter, gaussian, obstacles, island)

Modules:
    environment: Environment parsing from scenario files.
    solvers: Eikonal equation numerical solvers.
    path_finder: Path reconstruction algorithms.
    index_functions: Refractive index generators.
    plotting: Visualization utilities.
    utils: Console and helper functions.
"""

from .environment import Environment
from .solvers import LaxFriedrichsSolver, EikonalSolver
from .path_finder import (
    PathFinder,
    solve_euler,
    solve_heun,
    compute_gradient,
    interpolate_gradient,
)
from .index_functions import (
    IndexConfig,
    create_uniform_index,
    create_diopter_index,
    create_gaussian_index,
    create_obstacle_index,
    create_island_index,
    create_custom_index,
    get_index_function,
)
from .plotting import (
    plot_contour_lines,
    plot_index_map,
    plot_gradient_field,
    plot_path,
    plot_comparison,
    plot_obstacles,
)
from .utils import Console, console

__all__ = [
    # Environment
    'Environment',
    # Solvers
    'LaxFriedrichsSolver',
    'EikonalSolver',
    # Path finding
    'PathFinder',
    'solve_euler',
    'solve_heun',
    'compute_gradient',
    'interpolate_gradient',
    # Index functions
    'IndexConfig',
    'create_uniform_index',
    'create_diopter_index',
    'create_gaussian_index',
    'create_obstacle_index',
    'create_island_index',
    'create_custom_index',
    'get_index_function',
    # Plotting
    'plot_contour_lines',
    'plot_index_map',
    'plot_gradient_field',
    'plot_path',
    'plot_comparison',
    'plot_obstacles',
    # Utils
    'Console',
    'console',
]

__version__ = "0.1.0"
