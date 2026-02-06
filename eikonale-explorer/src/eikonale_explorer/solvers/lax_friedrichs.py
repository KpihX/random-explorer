import numpy as np
from tqdm import tqdm
from typing import Tuple

from .base import EikonalSolver
from ..utils import console

class LaxFriedrichsSolver(EikonalSolver):
    """
    Solves the Eikonal equation using the Lax-Friedrichs numerical scheme.
    Iteratively updates the potential function phi until convergence.
    """
    
    def __init__(self, max_iter: int = 5000, tol: float = 1e-9):
        self.max_iter = max_iter
        self.tol = tol
        
    def _hamiltonian(self, phi_x: np.ndarray, phi_y: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Hamiltonian for the Eikonal equation: sqrt(grad_phi^2) - N"""
        return np.sqrt(phi_x**2 + phi_y**2) - N
        
    def solve(self, N_map: np.ndarray, start_pos: Tuple[float, float], h: float, show_progress: bool = True) -> np.ndarray:
        """
        Execute the Lax-Friedrichs iteration.
        
        Args:
            N_map: Refractive index map (ny, nx).
            start_pos: (x, y) start.
            h: Grid step size.
            show_progress: Display tqdm bar.
        """
        ny, nx = N_map.shape
        dt = h / 2.0  # CFL condition usually h/sqrt(2) or h/2
        
        # Initialize phi with large values (infinity approx)
        # But start point is 0
        phi = np.full((ny, nx), 1e6) # Large value
        
        # Grid coordinates
        # start_pos is in [0, width] x [0, height]
        # Map to indices
        j_start = int(start_pos[0] / h)
        i_start = int(start_pos[1] / h)
        
        # Clip to bounds
        j_start = np.clip(j_start, 0, nx - 1)
        i_start = np.clip(i_start, 0, ny - 1)
        
        # Fix the boundary condition at the source
        phi[i_start, j_start] = 0.0
        
        iterator = tqdm(range(self.max_iter), disable=not show_progress, desc="Lax-Friedrichs Eikonal")
        
        for k in iterator:
            phi_old = phi.copy()
            
            # Central finite differences for gradients (with periodic/neumann handling)
            # Implementing Neumann (zero flux) at boundaries by padding/clipping
            
            # Vectorized neighbor access using rolling/shifting
            # North, South, East, West
            phi_N = np.roll(phi, -1, axis=0) # i+1
            phi_S = np.roll(phi, 1, axis=0)  # i-1
            phi_E = np.roll(phi, -1, axis=1) # j+1
            phi_W = np.roll(phi, 1, axis=1)  # j-1
            
            # Boundary conditions: copy edge values (Neumann-ish)
            # Roll wraps around, so we fix edges
            phi_N[-1, :] = phi[-1, :]
            phi_S[0, :] = phi[0, :]
            phi_E[:, -1] = phi[:, -1]
            phi_W[:, 0] = phi[:, 0]
            
            # Centered differences
            D_x = (phi_E - phi_W) / (2 * h)
            D_y = (phi_N - phi_S) / (2 * h)
            
            # Lax-Friedrichs Update step
            # phi_new = 0.25 * (N+S+E+W) - dt * H(Dx, Dy, N)
            
            # According to standard LF for Hamilton-Jacobi:
            # phi^{n+1} = (phi_E + phi_W + phi_N + phi_S)/4 - dt * H(Dx, Dy)
            # Note: We seek H=0. 
            # The iterative scheme to solve H(grad u) = 0 is essentially pseudo-time marching:
            # dphi/dt + H = 0  => phi^{n+1} = phi^n - dt * H
            # But with LF spatial averaging for stability.
            
            H_val = self._hamiltonian(D_x, D_y, N_map)
             
            # Update
            phi_new = 0.25 * (phi_N + phi_S + phi_E + phi_W) - dt * H_val
            
            # Re-impose source condition
            phi_new[i_start, j_start] = 0.0
            
            phi = phi_new
            
            # Convergence check
            diff = np.max(np.abs(phi - phi_old))
            if diff < self.tol:
                if show_progress:
                    console.print_success(f"Converged difference {diff:.2e} < {self.tol} at iter {k}")
                break
                
        return phi
