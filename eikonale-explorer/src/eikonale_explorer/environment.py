"""Environment parsing and management.

Parses scenario files in the same format as random-explorer.

Scenario file format (all values as floats, any whitespace separator):
    [0-1]   : x_max, y_max (domain dimensions)
    [2-3]   : start_x, start_y
    [4-5]   : goal_x, goal_y
    [6-7]   : start2_x, start2_y (multi-robot, unused here)
    [8-9]   : goal2_x, goal2_y (multi-robot, unused here)
    [10]    : radius (multi-robot, unused here)
    [11+]   : obstacles as (x, y, width, height) groups
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .utils import console


# Minimum number of values required in scenario file
MIN_NUM_VALUES = 11


class Environment:
    """Environment manager for Eikonal path planning.
    
    Attributes:
        width: Domain width (x_max).
        height: Domain height (y_max).
        start: Start position (x, y).
        goal: Goal position (x, y).
        obstacles: List of (x, y, w, h) rectangles.
    """
    
    def __init__(self, file_path: str):
        """Initialize environment from scenario file.
        
        Args:
            file_path: Path to scenario .txt file.
        """
        self.file_path = Path(file_path)
        self.width: float = 0.0
        self.height: float = 0.0
        self.start: Tuple[float, float] = (0.0, 0.0)
        self.goal: Tuple[float, float] = (0.0, 0.0)
        self.obstacles: List[Tuple[float, float, float, float]] = []
        
        # Multi-robot fields (kept for compatibility, unused in Eikonal)
        self.start2: Tuple[float, float] = (0.0, 0.0)
        self.goal2: Tuple[float, float] = (0.0, 0.0)
        self.radius: float = 0.0
        
        self._parse_file()
    
    def _parse_file(self):
        """Parse scenario file - same format as random-explorer."""
        path = self.file_path
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data_str = f.read()
        except FileNotFoundError:
            console.display_error(f"File not found: {path}")
            raise
        
        # Extract all numbers from file (any whitespace separator)
        tokens = data_str.replace('\n', ' ').split()
        values = [float(x) for x in tokens if x.strip()]
        
        if len(values) < MIN_NUM_VALUES:
            console.display_error(
                f"File has {len(values)} values, need at least {MIN_NUM_VALUES}"
            )
            raise ValueError(f"Insufficient data in file: {path}")
        
        # Parse fixed values (same order as random-explorer)
        self.width = values[0]   # x_max
        self.height = values[1]  # y_max
        self.start = (values[2], values[3])
        self.goal = (values[4], values[5])
        self.start2 = (values[6], values[7])  # multi-robot
        self.goal2 = (values[8], values[9])   # multi-robot
        self.radius = values[10]              # multi-robot
        
        # Parse obstacles (groups of 4)
        rest = values[MIN_NUM_VALUES:]
        if len(rest) % 4 != 0:
            console.display_error("Obstacle data must be a multiple of 4")
            raise ValueError("Invalid obstacle data")
        
        for i in range(0, len(rest), 4):
            obs = (rest[i], rest[i + 1], rest[i + 2], rest[i + 3])
            self.obstacles.append(obs)
        
        console.print_success(
            f"Parsed: {path.name} ({len(self.obstacles)} obstacles, "
            f"{self.width}x{self.height})"
        )
    
    def get_refractive_index_map(
        self,
        nx: int,
        ny: int,
        base_index: float = 1.0,
        obstacle_index: float = 1e5
    ) -> np.ndarray:
        """Generate discrete refractive index map.
        
        The map is normalized to [0,1] x [0,1] domain for the solver.
        
        Args:
            nx, ny: Grid dimensions.
            base_index: Index in free space.
            obstacle_index: Index inside obstacles (high = slow/impassable).
            
        Returns:
            2D array of shape (ny, nx) with index values.
        """
        # Create meshgrid in normalized [0,1] coordinates
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Initialize with base index
        N_map = np.full((ny, nx), base_index)
        
        # Add obstacles (normalized coordinates)
        for (ox, oy, w, h) in self.obstacles:
            # Normalize obstacle bounds
            ox_norm = ox / self.width
            oy_norm = oy / self.height
            w_norm = w / self.width
            h_norm = h / self.height
            
            mask = (
                (X >= ox_norm) & (X <= ox_norm + w_norm) &
                (Y >= oy_norm) & (Y <= oy_norm + h_norm)
            )
            N_map[mask] = obstacle_index
        
        return N_map
    
    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        title: Optional[str] = None,
        show_grid: bool = False
    ) -> plt.Axes:
        """Plot environment with obstacles, start, and goal.
        
        Args:
            ax: Matplotlib axes (created if None).
            show: Whether to call plt.show().
            title: Plot title.
            show_grid: Whether to show grid.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw obstacles
        for (ox, oy, w, h) in self.obstacles:
            rect = Rectangle(
                (ox, oy), w, h,
                facecolor='gray',
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Mark start and goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=12, 
                label=f'Start {self.start}')
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=15, 
                label=f'Goal {self.goal}')
        
        # Axes setup
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Environment: {self.file_path.name}")
        
        if show:
            plt.show()
        
        return ax
    
    def __repr__(self) -> str:
        return (
            f"Environment(file={self.file_path.name}, "
            f"size={self.width}x{self.height}, "
            f"obstacles={len(self.obstacles)})"
        )
