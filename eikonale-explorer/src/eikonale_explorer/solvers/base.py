from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
from ..utils import console

class EikonalSolver(ABC):
    """Abstract base class for Eikonal equation solvers."""
    
    @abstractmethod
    def solve(self, N_map: np.ndarray, start_pos: Tuple[float, float], h: float) -> np.ndarray:
        """
        Solve the Eikonal equation |grad(phi)| = N.
        
        Args:
            N_map: 2D array of refractive indices.
            start_pos: (x, y) starting position in physical coordinates.
            h: Grid spacing step.
            
        Returns:
            2D array representing the value function (phi).
        """
        pass
