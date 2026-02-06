from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ..environment import Environment
from ..solvers import LaxFriedrichsSolver
from ..path_finder import PathFinder
from ..utils import console

def main(file_path: str, grid_size: int = 128, max_iter: int = 5000, output_path: str = None):
    """Main solver pipeline."""
    console.panel(f"Solving Eikonal for {file_path}", title="Startup")
    
    # 1. Load Environment
    env = Environment(file_path)
    
    # 2. Discrete Grid & Refractive Index
    h = env.width / grid_size
    nx = int(env.width / h)
    ny = int(env.height / h)
    
    console.info(f"Grid: {nx}x{ny} (h={h:.4f})")
    
    # Generate Index Map (Cost)
    # 1.0 in free space, 100.0 in obstacles
    N_map = env.get_refractive_index_map(nx, ny, base_index=1.0, obstacle_index=100.0)
    
    # 3. Solve Eikonal Equation (Value Function)
    solver = LaxFriedrichsSolver(max_iter=max_iter, tol=1e-6)
    console.info("Running Lax-Friedrichs Solver...")
    
    phi = solver.solve(N_map, env.start, h)
    
    # 4. Path Reconstruction (Gradient Descent)
    console.info("Reconstructing path (Heun's method)...")
    path = PathFinder.solve_heun(phi, env.start, env.goal, h)
    
    # 5. Authorization / Validation (Did we hit start?)
    # Since we backtracked from Goal, path[-1] should be near Start
    # Wait: solve_heun returns path from START to GOAL (reversed internally)
    dist_error = np.linalg.norm(path[0] - env.start)
    if dist_error > h * 2:
        console.warning(f"Path start deviation: {dist_error:.2f}")
    
    # 6. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Environment Obstacles
    env.plot(ax=ax, show=False)
    
    # Plot Contours of Phi
    # Create meshgrid for plotting (x, y physical coords)
    x = np.linspace(0, env.width, nx)
    y = np.linspace(0, env.height, ny)
    X, Y = np.meshgrid(x, y)
    
    # Use contours
    # Mask high values inside obstacles where phi might not have converged or is huge
    phi_display = np.ma.masked_where(N_map > 10, phi)
    
    cs = ax.contour(X, Y, phi_display, levels=30, cmap='viridis', alpha=0.5)
    plt.colorbar(cs, ax=ax, label='Time / Cost ($\phi$)')
    
    # Plot Path
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Eikonal Path')
        console.success(f"Path found! Length: {len(path)} steps.")
    else:
        console.error("Path reconstruction failed.")
        
    ax.legend()
    ax.set_title(f"Eikonal Path Planning - {Path(file_path).name}")
    
    if output_path:
        plt.savefig(output_path)
        console.success(f"Saved plot to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
