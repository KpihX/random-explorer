from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .utils import console

@dataclass
class Environment:
    """
    Represents the 2D environment with parsing from scenario files.
    Adapts discrete obstacles to continuous refractive index maps for Eikonal solvers.
    """
    width: float
    height: float
    start: Tuple[float, float]
    goal: Tuple[float, float]
    # start2/goal2 for multi-robot scenarios (kept for compatibility)
    start2: Optional[Tuple[float, float]] = None
    goal2: Optional[Tuple[float, float]] = None
    obstacles: List[Tuple[float, float, float, float]] = None  # (x, y, w, h)
    
    def __init__(self, file_path: str):
        self.obstacles = []
        self._parse_file(file_path)
        
    def _parse_file(self, file_path: str):
        """Parse scenario file format."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")
            
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        try:
            self.width = float(lines[0])
            self.height = float(lines[1])
            self.start = (float(lines[2]), float(lines[3]))
            self.goal = (float(lines[4]), float(lines[5]))
            
            # Check for second robot (lines 6-9 usually)
            # The format seems to vary slightly, but assuming standard format:
            self.start2 = (float(lines[6]), float(lines[7]))
            self.goal2 = (float(lines[8]), float(lines[9]))
            
            # Line 10 (index 10) is typically R (radius) or similar
            # Obstacles start from line 11 (index 11)
            raw_obstacles = lines[11:]
            
            for line in raw_obstacles:
                parts = list(map(float, line.split()))
                if len(parts) >= 4:
                    self.obstacles.append((parts[0], parts[1], parts[2], parts[3]))
                    
            console.print_success(f"Parsed environment from {path.name}: {len(self.obstacles)} obstacles.")
            
        except (ValueError, IndexError) as e:
            console.display_error(f"Error parsing file {file_path}: {e}")
            raise

    def get_refractive_index_map(self, nx: int, ny: int, base_index: float = 1.0, obstacle_index: float = 100.0) -> np.ndarray:
        """
        Generate a discrete refractive index map N(i,j).
        
        Args:
            nx: Number of grid points in X.
            ny: Number of grid points in Y.
            base_index: Index in free space (default 1.0).
            obstacle_index: Index inside obstacles (default 100.0, high cost).
            
        Returns:
            2D numpy array of shape (ny, nx) containing the refractive index.
        """
        x = np.linspace(0, self.width, nx)
        y = np.linspace(0, self.height, ny)
        xv, yv = np.meshgrid(x, y)
        
        # Start with uniform index
        N_map = np.full((ny, nx), base_index)
        
        # Apply obstacles
        # Note: A point is in a rectangle (x0, y0, w, h) if x0 <= x <= x0+w and y0 <= y <= y0+h
        for (ox, oy, w, h) in self.obstacles:
            mask = (xv >= ox) & (xv <= ox + w) & (yv >= oy) & (yv <= oy + h)
            N_map[mask] = obstacle_index
            
        return N_map

    def plot(self, ax: plt.Axes = None, show: bool = False, title: str = "Environment"):
        """Visualize the environment with standard matplotlib patches."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        # Limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # Obstacles
        for (x, y, w, h) in self.obstacles:
            rect = Rectangle((x, y), w, h, color='black', alpha=0.6)
            ax.add_patch(rect)
            
        # Start/Goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'rx', markersize=10, label='Goal')
        
        if self.start2:
             ax.plot(self.start2[0], self.start2[1], 'go', alpha=0.5, markersize=8)
        if self.goal2:
             ax.plot(self.goal2[0], self.goal2[1], 'rx', alpha=0.5, markersize=8)
             
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend()
        
        if show:
            plt.show()
        return ax
