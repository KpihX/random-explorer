from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm
from .utils import console

class PathFinder:
    """
    Reconstructs the shortest path by following the gradient of the value function phi.
    Solves the ODE: X'(t) = -grad(phi)(X(t)) / N^2(X(t))  (or simply descending phi)
    """
    
    @staticmethod
    def _grad_phi(phi: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute discrete gradient of phi using central differences."""
        # Using np.gradient which handles boundaries automatically (2nd order inside, 1st order edge)
        grad_xy = np.gradient(phi, h)
        # np.gradient returns [grad_axis0, grad_axis1] -> [grad_y, grad_x]
        return grad_xy[1], grad_xy[0] 

    @staticmethod
    def _interpolate_grad(x: float, y: float, 
                          grad_x: np.ndarray, grad_y: np.ndarray, 
                          h: float, nx: int, ny: int) -> Tuple[float, float]:
        """Bilinear interpolation of gradient at (x, y)."""
        # Convert physical x, y to grid indices (float)
        j = x / h
        i = y / h
        
        # Determine surrounding indices
        j0 = int(np.floor(j))
        j1 = min(j0 + 1, nx - 1)
        i0 = int(np.floor(i))
        i1 = min(i0 + 1, ny - 1)
        j0 = max(0, min(j0, nx - 1))
        
        # Weights
        tx = j - j0
        ty = i - i0
        
        # Interpolate Gx
        # f(x,y) ~ (1-tx)(1-ty)f00 + tx(1-ty)f10 + (1-tx)ty f01 + tx ty f11
        gx00 = grad_x[i0, j0]
        gx10 = grad_x[i0, j1]
        gx01 = grad_x[i1, j0]
        gx11 = grad_x[i1, j1]
        
        gx = (1 - tx) * (1 - ty) * gx00 + \
             tx * (1 - ty) * gx10 + \
             (1 - tx) * ty * gx01 + \
             tx * ty * gx11
             
        # Interpolate Gy
        gy00 = grad_y[i0, j0]
        gy10 = grad_y[i0, j1]
        gy01 = grad_y[i1, j0]
        gy11 = grad_y[i1, j1]
        
        gy = (1 - tx) * (1 - ty) * gy00 + \
             tx * (1 - ty) * gy10 + \
             (1 - tx) * ty * gy01 + \
             tx * ty * gy11
             
        return gx, gy

    @classmethod
    def solve_euler(cls, phi: np.ndarray, N_map: np.ndarray, start: Tuple[float, float], goal: Tuple[float, float], h: float, dt: float = None, max_steps: int = 10000, tol: float = 1e-2) -> np.ndarray:
        """
        Reconstruct path using Euler method. 
        Note: We go from Start to Goal if phi is computed from Goal, 
              or Goal to Start if phi is computed from Start.
        Usually Eikonal solves 'time to reach target', so we descend from Start to Target (where phi=0).
        Here we assume phi is computed from SOURCE (Start), so gradient points AWAY.
        To find path from Source to Destination:
           The characteristics (rays) follow grad(phi).
        WAIT: If phi is "Time from Source", then gradient points in direction of propagation.
              So we just integrate X' = grad(phi)/N^2.
        
        HOWEVER, typically to find geodesics between A and B:
        1. Compute phi(x) = distance from A.
        2. Start at B, follow -grad(phi) back to A.
        
        Let's support "Backtracking" mode (Goal -> Start).
        """
        if dt is None:
            dt = h / 2.0
            
        ny, nx = phi.shape
        grad_x, grad_y = cls._grad_phi(phi, h)
        
        # Normalize gradient for better stability or follow strictly Eikonal flow
        # Flow: dX/ds = grad(phi) / |grad(phi)| (unit speed)
        # Using Euler: X_{n+1} = X_n - dt * grad(phi) (descent)
        
        path = [goal] # Start backtracking from Goal
        current_pos = np.array(goal)
        
        target = np.array(start)
        
        for _ in range(max_steps):
            gx, gy = cls._interpolate_grad(current_pos[0], current_pos[1], grad_x, grad_y, h, nx, ny)

            # Gradient Descent: Move Opposite to Gradient of Distance from Start
            # Because phi increases away from start. To go back to start, we go against gradient.
            
            # Normalize to unit step for robustness
            norm = np.hypot(gx, gy)
            if norm < 1e-6:
                break
                
            dx = - (gx / norm) # Direction
            
            current_pos = current_pos + dx * dt
            path.append(current_pos.copy())
            
            if np.linalg.norm(current_pos - target) < tol:
                console.success("Euler path reconstruction reached target.")
                break
                
            # Bounds check
            if not (0 <= current_pos[0] <= (nx-1)*h and 0 <= current_pos[1] <= (ny-1)*h):
                console.warning("Path went out of bounds.")
                break
                
        return np.array(path)[::-1] # Reverse to get Start -> Goal

    @classmethod
    def solve_heun(cls, phi: np.ndarray, start: Tuple[float, float], goal: Tuple[float, float], h: float, dt: float = None, max_steps: int = 10000, tol: float = 1e-2) -> np.ndarray:
        """Reconstruct path using Heun's method (Runge-Kutta 2)."""
        if dt is None:
            dt = h / 2.0
            
        ny, nx = phi.shape
        grad_x, grad_y = cls._grad_phi(phi, h)
        
        path = [goal]
        current_pos = np.array(goal)
        target = np.array(start)
        
        for _ in range(max_steps):
            # Step 1: Endpoint predictor (Euler)
            gx1, gy1 = cls._interpolate_grad(current_pos[0], current_pos[1], grad_x, grad_y, h, nx, ny)
            norm1 = np.hypot(gx1, gy1)
            if norm1 < 1e-6: break
            
            d1 = - np.array([gx1, gy1]) / norm1
            
            intermediate_pos = current_pos + d1 * dt
            
             # Check bounds for intermediate
            if not (0 <= intermediate_pos[0] <= (nx-1)*h and 0 <= intermediate_pos[1] <= (ny-1)*h):
                break

            # Step 2: Corrector
            gx2, gy2 = cls._interpolate_grad(intermediate_pos[0], intermediate_pos[1], grad_x, grad_y, h, nx, ny)
            norm2 = np.hypot(gx2, gy2)
            if norm2 < 1e-6: 
                d2 = d1 # Fallback
            else:
                d2 = - np.array([gx2, gy2]) / norm2
                
            # Heun update
            current_pos = current_pos + (d1 + d2) / 2.0 * dt
            path.append(current_pos.copy())
            
            if np.linalg.norm(current_pos - target) < tol:
                console.success("Heun path reconstruction reached target.")
                break
                
        return np.array(path)[::-1]
