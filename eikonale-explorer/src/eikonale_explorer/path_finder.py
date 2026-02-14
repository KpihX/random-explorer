"""Path reconstruction from Eikonal solution.

This module provides algorithms to extract the shortest path from
the potential field φ by following the gradient descent.

Methods:
- Euler explicit: First order, simple but less accurate
- Heun (RK2): Second order, more accurate

The path goes from goal to source (backtracking) following -∇φ.
"""

from typing import Tuple, List, Optional
import numpy as np
from .utils import console


# =============================================================================
# Gradient Computation
# =============================================================================

def compute_gradient(phi: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradient of phi using central differences.
    
    Uses numpy.gradient which handles boundaries automatically
    with second-order accuracy in interior and first-order at edges.
    
    Args:
        phi: 2D potential field.
        h: Grid spacing.
        
    Returns:
        (grad_x, grad_y): Partial derivatives.
    """
    # np.gradient returns [grad_axis0, grad_axis1] = [grad_y, grad_x]
    grads = np.gradient(phi, h)
    grad_y = grads[0]  # derivative along axis 0 (rows = y direction)
    grad_x = grads[1]  # derivative along axis 1 (cols = x direction)
    return grad_x, grad_y


# =============================================================================
# Index Finding (for interpolation)
# =============================================================================

def find_cell_indices(
    x: float, 
    y: float, 
    mesh_x: np.ndarray, 
    mesh_y: np.ndarray
) -> Tuple[int, int]:
    """Find grid cell indices containing point (x, y).
    
    Finds i, j such that mesh_x[i] <= x < mesh_x[i+1]
    and mesh_y[j] <= y < mesh_y[j+1].
    
    Args:
        x, y: Point coordinates.
        mesh_x, mesh_y: Grid coordinate arrays.
        
    Returns:
        (i, j): Cell indices.
    """
    # Find index of closest grid point <= coordinate
    i = np.searchsorted(mesh_x, x, side='right') - 1
    j = np.searchsorted(mesh_y, y, side='right') - 1
    
    # Clamp to valid range
    i = np.clip(i, 0, len(mesh_x) - 2)
    j = np.clip(j, 0, len(mesh_y) - 2)
    
    return int(i), int(j)


# =============================================================================
# Gradient Interpolation
# =============================================================================

def interpolate_gradient(
    x: float,
    y: float,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    h: float,
    nx: int,
    ny: int
) -> Tuple[float, float]:
    """Bilinear interpolation of gradient at arbitrary point.
    
    Given discrete gradient values on grid, interpolate to get
    gradient at any point (x, y) using the four nearest neighbors.
    
    Args:
        x, y: Point coordinates (in [0, domain_size]).
        grad_x, grad_y: Discrete gradient arrays.
        h: Grid spacing.
        nx, ny: Grid dimensions.
        
    Returns:
        (gx, gy): Interpolated gradient.
    """
    # Convert to fractional grid indices
    fi = x / h
    fj = y / h
    
    # Integer indices of bottom-left corner
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    
    # Clamp to grid
    i0 = np.clip(i0, 0, nx - 2)
    j0 = np.clip(j0, 0, ny - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    
    # Interpolation weights
    tx = fi - i0
    ty = fj - j0
    
    # Bilinear interpolation for grad_x
    # f(x,y) ≈ (1-tx)(1-ty)f00 + tx(1-ty)f10 + (1-tx)ty f01 + tx ty f11
    gx00 = grad_x[j0, i0]
    gx10 = grad_x[j0, i1]
    gx01 = grad_x[j1, i0]
    gx11 = grad_x[j1, i1]
    
    gx = (1 - tx) * (1 - ty) * gx00 + \
         tx * (1 - ty) * gx10 + \
         (1 - tx) * ty * gx01 + \
         tx * ty * gx11
    
    # Bilinear interpolation for grad_y
    gy00 = grad_y[j0, i0]
    gy10 = grad_y[j0, i1]
    gy01 = grad_y[j1, i0]
    gy11 = grad_y[j1, i1]
    
    gy = (1 - tx) * (1 - ty) * gy00 + \
         tx * (1 - ty) * gy10 + \
         (1 - tx) * ty * gy01 + \
         tx * ty * gy11
    
    return float(gx), float(gy)


# =============================================================================
# Euler Explicit Method
# =============================================================================

def solve_euler(
    phi: np.ndarray,
    N_map: np.ndarray,
    source: Tuple[float, float],
    goal: Tuple[float, float],
    h: float,
    dt: Optional[float] = None,
    max_steps: int = 10000,
    tol: float = 1e-3
) -> np.ndarray:
    """Reconstruct path using Euler explicit method.
    
    Solves the ODE: dX/dt = -∇φ/|∇φ| (normalized gradient descent)
    Starting from goal, backtrack to source.
    
    Args:
        phi: Potential field (computed from Eikonal).
        N_map: Refractive index map (for normalization if needed).
        source: Source point (where φ=0).
        goal: Destination point (start of backtracking).
        h: Grid spacing.
        dt: Time step (default: h/2).
        max_steps: Maximum iterations.
        tol: Tolerance for reaching source.
        
    Returns:
        Path array of shape (n_points, 2), ordered source -> goal.
    """
    if dt is None:
        dt = h / 2.0
    
    ny, nx = phi.shape
    grad_x, grad_y = compute_gradient(phi, h)
    
    # Start from goal, backtrack to source
    path = [np.array(goal)]
    current = np.array(goal, dtype=float)
    target = np.array(source)
    
    for step in range(max_steps):
        # Interpolate gradient at current position
        gx, gy = interpolate_gradient(
            current[0], current[1], grad_x, grad_y, h, nx, ny
        )
        
        # Normalize gradient
        norm = np.hypot(gx, gy)
        if norm < 1e-10:
            console.print_warning(f"Euler: Gradient vanished at step {step}")
            break
        
        # Move opposite to gradient (descend potential)
        direction = -np.array([gx, gy]) / norm
        
        # Euler step
        current = current + dt * direction
        path.append(current.copy())
        
        # Check if reached source
        if np.linalg.norm(current - target) < tol:
            console.print_success(f"Euler: Reached source in {step+1} steps")
            break
        
        # Bounds check
        domain_max = (nx - 1) * h
        if not (0 <= current[0] <= domain_max and 0 <= current[1] <= domain_max):
            console.print_warning(f"Euler: Path left domain at step {step}")
            break
    
    # Reverse to get source -> goal order
    return np.array(path[::-1])


# =============================================================================
# Heun Method (Runge-Kutta 2)
# =============================================================================

def solve_heun(
    phi: np.ndarray,
    source: Tuple[float, float],
    goal: Tuple[float, float],
    h: float,
    dt: Optional[float] = None,
    max_steps: int = 10000,
    tol: float = 1e-3
) -> np.ndarray:
    """Reconstruct path using Heun's method (improved Euler / RK2).
    
    Heun's method uses a predictor-corrector approach:
    1. Predictor: X̃ = X + dt * f(X)
    2. Corrector: X_new = X + dt/2 * (f(X) + f(X̃))
    
    This gives second-order accuracy.
    
    Args:
        phi: Potential field.
        source: Source point.
        goal: Goal point.
        h: Grid spacing.
        dt: Time step (default: h/2).
        max_steps: Maximum iterations.
        tol: Tolerance for reaching source.
        
    Returns:
        Path array of shape (n_points, 2), ordered source -> goal.
    """
    if dt is None:
        dt = h / 2.0
    
    ny, nx = phi.shape
    grad_x, grad_y = compute_gradient(phi, h)
    
    path = [np.array(goal)]
    current = np.array(goal, dtype=float)
    target = np.array(source)
    domain_max = (nx - 1) * h
    
    for step in range(max_steps):
        # === Step 1: Predictor (Euler step) ===
        gx1, gy1 = interpolate_gradient(
            current[0], current[1], grad_x, grad_y, h, nx, ny
        )
        norm1 = np.hypot(gx1, gy1)
        if norm1 < 1e-10:
            console.print_warning(f"Heun: Gradient vanished at step {step}")
            break
        
        d1 = -np.array([gx1, gy1]) / norm1
        intermediate = current + dt * d1
        
        # Bounds check for intermediate
        if not (0 <= intermediate[0] <= domain_max and 0 <= intermediate[1] <= domain_max):
            # Fallback to Euler step
            current = current + dt * d1
            path.append(current.copy())
            continue
        
        # === Step 2: Corrector ===
        gx2, gy2 = interpolate_gradient(
            intermediate[0], intermediate[1], grad_x, grad_y, h, nx, ny
        )
        norm2 = np.hypot(gx2, gy2)
        
        if norm2 < 1e-10:
            d2 = d1  # Fallback
        else:
            d2 = -np.array([gx2, gy2]) / norm2
        
        # Heun update: average of both directions
        current = current + dt * (d1 + d2) / 2.0
        path.append(current.copy())
        
        # Check if reached source
        if np.linalg.norm(current - target) < tol:
            console.print_success(f"Heun: Reached source in {step+1} steps")
            break
    
    return np.array(path[::-1])


# =============================================================================
# PathFinder Class (unified interface)
# =============================================================================

class PathFinder:
    """Unified interface for path reconstruction methods."""
    
    METHODS = {
        'euler': solve_euler,
        'heun': solve_heun,
    }
    
    @classmethod
    def solve(
        cls,
        phi: np.ndarray,
        source: Tuple[float, float],
        goal: Tuple[float, float],
        h: float,
        method: str = 'heun',
        **kwargs
    ) -> np.ndarray:
        """Solve path reconstruction using specified method.
        
        Args:
            phi: Potential field.
            source, goal: Endpoints.
            h: Grid spacing.
            method: 'euler' or 'heun'.
            **kwargs: Additional method arguments.
            
        Returns:
            Path array.
        """
        if method not in cls.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(cls.METHODS.keys())}")
        
        solver = cls.METHODS[method]
        
        if method == 'euler':
            # Euler needs N_map, we'll pass None and let it handle
            return solver(phi, None, source, goal, h, **kwargs)
        else:
            return solver(phi, source, goal, h, **kwargs)
    
    # Convenience class methods for direct access
    @staticmethod
    def solve_euler(phi, N_map, source, goal, h, **kwargs):
        return solve_euler(phi, N_map, source, goal, h, **kwargs)
    
    @staticmethod
    def solve_heun(phi, source, goal, h, **kwargs):
        return solve_heun(phi, source, goal, h, **kwargs)
