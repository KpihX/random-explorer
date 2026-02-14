"""Solve command for Eikonal CLI.

Orchestrates the full pipeline:
1. Load environment from scenario file
2. Generate refractive index map from obstacles
3. Solve Eikonal equation
4. Reconstruct optimal path
5. Visualize results
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from ..environment import Environment
from ..solvers import LaxFriedrichsSolver
from ..path_finder import PathFinder
from ..plotting import plot_contour_lines, plot_obstacles
from ..utils import console


def main(
    file_path: str,
    grid_size: int = 128,
    max_iter: int = 5000,
    output_path: Optional[str] = None
):
    """Main solver pipeline.
    
    Args:
        file_path: Path to scenario file.
        grid_size: Grid resolution (assumes square domain).
        max_iter: Maximum solver iterations.
        output_path: Optional path to save output image.
    """
    console.display(f"[bold]Eikonale Explorer[/bold] - Solving {file_path}", 
                    title="🚀 Startup", border_style="blue")
    
    # 1. Load Environment
    env = Environment(file_path)
    
    # 2. Compute grid parameters
    h = env.width / grid_size
    nx = int(env.width / h)
    ny = int(env.height / h)
    
    console.print_info(f"Grid: {nx}x{ny} (h={h:.4f})")
    console.print_info(f"Source: {env.start}, Goal: {env.goal}")
    console.print_info(f"Obstacles: {len(env.obstacles)}")
    
    # 3. Generate Index Map
    N_map = env.get_refractive_index_map(nx, ny, 
                                          base_index=1.0, 
                                          obstacle_index=1e5)
    
    # Normalize source/goal to [0,1] domain
    source_norm = (env.start[0] / env.width, env.start[1] / env.height)
    goal_norm = (env.goal[0] / env.width, env.goal[1] / env.height)
    h_norm = 1.0 / nx
    
    # 4. Solve Eikonal Equation
    console.print_info("Running Lax-Friedrichs Solver...")
    solver = LaxFriedrichsSolver(max_iter=max_iter, tol=1e-6)
    phi = solver.solve(N_map, source_norm, h_norm, show_progress=True)
    
    # 5. Path Reconstruction
    console.print_info("Reconstructing path (Heun's method)...")
    path_norm = PathFinder.solve_heun(phi, source_norm, goal_norm, h_norm)
    
    # De-normalize path to original coordinates
    path = path_norm * np.array([env.width, env.height])
    
    # 6. Validation
    if len(path) > 0:
        start_error = np.linalg.norm(path[0] - np.array(env.start))
        goal_error = np.linalg.norm(path[-1] - np.array(env.goal))
        
        if start_error > h * 3:
            console.print_warning(f"Path start deviation: {start_error:.2f}")
        if goal_error > h * 3:
            console.print_warning(f"Path goal deviation: {goal_error:.2f}")
    
    # 7. Visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot environment (obstacles, start, goal)
    env.plot(ax=ax, show=False)
    
    # Overlay phi contours (scaled to environment coordinates)
    x = np.linspace(0, env.width, nx)
    y = np.linspace(0, env.height, ny)
    X, Y = np.meshgrid(x, y)
    
    phi_masked = np.ma.masked_where(N_map > 100, phi)
    cs = ax.contour(X, Y, phi_masked, levels=30, cmap='viridis', alpha=0.4)
    
    # Plot path
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2.5, label='Eikonal Path')
        console.print_success(f"Path found! {len(path)} steps, length: {compute_path_length(path):.2f}")
    else:
        console.display_error("Path reconstruction failed.")
    
    ax.legend()
    ax.set_title(f"Eikonal Path Planning — {Path(file_path).name}")
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print_success(f"Saved plot to {output_path}")
    else:
        plt.show()


def compute_path_length(path: np.ndarray) -> float:
    """Compute total path length."""
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
