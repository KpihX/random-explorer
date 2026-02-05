"""Environment module for path planning problems.

This module provides the Environment class that handles parsing scenario files,
collision detection, and visualization of the path planning environment.
"""

from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .utils import Console


# Type aliases
Point = Tuple[float, float]
Obstacle = Tuple[float, float, float, float]  # (x, y, width, height)
Path = List[Point]


class Environment:
    """Represents a 2D path planning environment with obstacles.
    
    The environment is a rectangle with obstacles defined as axis-aligned
    rectangles. It supports collision detection and visualization.
    
    Attributes:
        x_max: Maximum x coordinate of the environment.
        y_max: Maximum y coordinate of the environment.
        start: Starting position (x, y) for robot 1.
        goal: Goal position (x, y) for robot 1.
        start2: Starting position (x, y) for robot 2 (multi-robot problems).
        goal2: Goal position (x, y) for robot 2 (multi-robot problems).
        radius: Safety radius for multi-robot problems.
        obstacles: List of obstacles as (x, y, width, height) tuples.
    """
    
    # Minimum number of values required in source file
    MIN_NUM_VALUES = 11
    
    def __init__(self, file_path: str) -> None:
        """Initialize environment from a scenario file.
        
        Args:
            file_path: Path to the scenario file.
            
        Raises:
            FileNotFoundError: If the scenario file does not exist.
            ValueError: If the file format is invalid or data is inconsistent.
        """
        # Initialize console for logging
        self.console = Console()
        
        # Initialize attributes
        self.x_max: float = 0.0
        self.y_max: float = 0.0
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.obstacles: List[Obstacle] = []
        
        # Parameters for multi-robot problems (Section 5)
        self.start2: Optional[Point] = None
        self.goal2: Optional[Point] = None
        self.radius: float = 0.0
        
        # Parse and validate
        self._parse_data(file_path)
        
        if not self._check_validity():
            self.console.display_error(
                "Environment data is invalid", "EnvironmentError"
            )
            raise ValueError("Invalid environment data")
        
        # self.console.display(
        #     "Environment initialized successfully",
        #     "Environment",
        #     border_style="green"
        # )
    
    def _parse_data(self, file_path: str) -> None:
        """Parse scenario file and extract environment data.
        
        Args:
            file_path: Path to the scenario file.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is invalid.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_str = f.read()
        except FileNotFoundError:
            self.console.display_error(
                f"Data file not found: {file_path}", "FileNotFoundError"
            )
            raise
        
        # Extract all numbers from file
        tokens = data_str.replace('\n', ' ').split()
        values = [float(x) for x in tokens if x.strip()]
        
        if len(values) < self.MIN_NUM_VALUES:
            self.console.display_error(
                f"File has {len(values)} values, need at least {self.MIN_NUM_VALUES}",
                "ValueError"
            )
            raise ValueError("Insufficient data in file")
        
        # Parse fixed values
        self.x_max = values[0]
        self.y_max = values[1]
        self.start = (values[2], values[3])
        self.goal = (values[4], values[5])
        self.start2 = (values[6], values[7])
        self.goal2 = (values[8], values[9])
        self.radius = values[10]
        
        # Parse obstacles
        rest = values[self.MIN_NUM_VALUES:]
        if len(rest) % 4 != 0:
            self.console.display_error(
                "Obstacle coordinates must be a multiple of 4",
                "ValueError"
            )
            raise ValueError("Invalid obstacle data")
        
        for i in range(0, len(rest), 4):
            obs: Obstacle = (rest[i], rest[i + 1], rest[i + 2], rest[i + 3])
            self.obstacles.append(obs)
    
    def is_collision(self, point: Point) -> bool:
        """Check if a point collides with the environment.
        
        A collision occurs if the point is outside the environment bounds
        or inside any obstacle (including borders).
        
        Args:
            point: The (x, y) coordinates to check.
            
        Returns:
            True if there is a collision, False otherwise.
        """
        x, y = point
        
        # Check environment bounds
        if x < 0 or x > self.x_max or y < 0 or y > self.y_max:
            return True
        
        # Check obstacles (borders included)
        for xo, yo, lx, ly in self.obstacles:
            if xo <= x <= xo + lx and yo <= y <= yo + ly:
                return True
        
        return False
    
    @staticmethod
    def compute_path_length(path: np.ndarray) -> float:
        """Compute the total Euclidean length of a path.
        
        Args:
            path: Array of shape (n, 2) representing waypoints.
            
        Returns:
            Total path length (sum of segment lengths).
        """
        if path is None or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(path[i + 1] - path[i])
        
        return total_length
    
    def check_segment_collision_hard(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        step: float = 10.0
    ) -> bool:
        """Check if a segment collides using point sampling (hard mode).
        
        Samples points along the segment and checks each for collision.
        
        Args:
            p1: Start point of segment.
            p2: End point of segment.
            step: Distance between sample points.
            
        Returns:
            True if any sampled point collides, False otherwise.
        """
        dist = np.linalg.norm(p2 - p1)
        if dist == 0:
            return self.is_collision(p1)
        
        # Sample points along segment
        n_samples = int(np.ceil(dist / step)) + 1
        samples = np.linspace(p1, p2, n_samples)
        
        for point in samples:
            if self.is_collision(point):
                return True
        return False
    
    def _segment_obstacle_overlap(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        obstacle: Obstacle
    ) -> float:
        """Calculate the overlap length between a segment and an obstacle.
        
        Uses parametric line intersection (Liang-Barsky algorithm) to compute
        the exact length of segment that lies within the obstacle.
        
        Args:
            p1: Start point of segment.
            p2: End point of segment.
            obstacle: Obstacle as (x, y, width, height).
            
        Returns:
            Length of segment inside the obstacle.
        """
        xo, yo, lx, ly = obstacle
        x1, y1 = xo + lx, yo + ly
        
        # Direction vector
        d = p2 - p1
        
        # Handle X axis
        if np.isclose(d[0], 0.0):  # Vertical segment
            if not (xo <= p1[0] <= x1):
                return 0.0
            t_x_enter, t_x_exit = 0.0, 1.0
        else:
            t1 = (xo - p1[0]) / d[0]
            t2 = (x1 - p1[0]) / d[0]
            t_x_enter = min(t1, t2)
            t_x_exit = max(t1, t2)
        
        # Handle Y axis
        if np.isclose(d[1], 0.0):  # Horizontal segment
            if not (yo <= p1[1] <= y1):
                return 0.0
            t_y_enter, t_y_exit = 0.0, 1.0
        else:
            t1 = (yo - p1[1]) / d[1]
            t2 = (y1 - p1[1]) / d[1]
            t_y_enter = min(t1, t2)
            t_y_exit = max(t1, t2)
        
        # Compute intersection interval
        t_entry = max(0.0, t_x_enter, t_y_enter)
        t_exit = min(1.0, t_x_exit, t_y_exit)
        
        if t_entry < t_exit:
            segment_len = np.linalg.norm(d)
            return (t_exit - t_entry) * segment_len
        
        return 0.0
    
    def check_segment_collision_soft(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate total collision length for a segment (soft mode).
        
        Computes the sum of overlap lengths with all obstacles using
        geometric intersection (no sampling needed).
        
        Args:
            p1: Start point of segment.
            p2: End point of segment.
            
        Returns:
            Total length of segment that overlaps with obstacles.
        """
        # Bounding box of segment for early rejection
        seg_min_x = min(p1[0], p2[0])
        seg_max_x = max(p1[0], p2[0])
        seg_min_y = min(p1[1], p2[1])
        seg_max_y = max(p1[1], p2[1])
        
        overlap_length = 0.0
        
        for obs in self.obstacles:
            xo, yo, lx, ly = obs
            
            # Early rejection using bounding box
            if (seg_max_x < xo or seg_min_x > xo + lx or
                seg_max_y < yo or seg_min_y > yo + ly):
                continue
            
            overlap_length += self._segment_obstacle_overlap(p1, p2, obs)
        
        return overlap_length
    
    def evaluate_path_collision(
        self,
        path: np.ndarray,
        soft_mode: bool = True
    ) -> float:
        """Evaluate collision metric for a complete path.
        
        This is a unified method for both hard and soft modes:
        - Soft mode: Returns total collision length (sum of overlap lengths)
        - Hard mode: Returns collision count (number of segments with collision)
        
        Args:
            path: Full path array of shape (n, 2) including start and goal.
            soft_mode: If True, return collision length; otherwise collision count.
            
        Returns:
            Collision metric (length for soft, count for hard). 0 means no collision.
        """
        if path is None or len(path) < 2:
            return 0.0
        
        collision_metric = 0.0
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            
            if soft_mode:
                # Soft: accumulate collision length
                collision_metric += self.check_segment_collision_soft(p1, p2)
            else:
                # Hard: count collisions
                collision_metric += self.check_segment_collision_hard(p1, p2)
        
        return collision_metric
    
    def _is_valid_obstacle(self, obstacle: Obstacle) -> bool:
        """Check if an obstacle is valid.
        
        An obstacle is valid if it has positive dimensions and fits
        entirely within the environment bounds.
        
        Args:
            obstacle: The (x, y, width, height) tuple to check.
            
        Returns:
            True if the obstacle is valid, False otherwise.
        """
        xo, yo, lx, ly = obstacle
        
        # Check positive dimensions
        if lx <= 0 or ly <= 0:
            return False
        
        # Check bounds
        if xo < 0 or yo < 0 or xo + lx > self.x_max or yo + ly > self.y_max:
            return False
        
        return True
    
    def _check_validity(self) -> bool:
        """Check if the environment configuration is valid.
        
        Returns:
            True if environment is valid, False otherwise.
        """
        # Validate all obstacles
        for obs in self.obstacles:
            if not self._is_valid_obstacle(obs):
                return False
        
        # Validate start/goal positions and radius
        if self.start is None or self.goal is None:
            return False
            
        return (
            self.radius > 0 and
            not self.is_collision(self.start) and
            not self.is_collision(self.goal)
        )
    
    def plot_environment(
        self,
        path: Optional[Path] = None,
        title: str = "Environment Plot",
        ax: Optional[plt.Axes] = None,
        path_color: str = 'blue',
        path_label: str = 'Path',
        show_legend: bool = True
    ) -> Tuple[Optional[plt.Figure], plt.Axes]:
        """Plot the environment with optional path overlay.
        
        Args:
            path: Optional list of (x, y) points defining a path.
            title: Title for the plot.
            ax: Optional matplotlib axes. If None, creates new figure.
            path_color: Color for the path ('blue', 'green', 'red', etc.).
            path_label: Label for the path in legend.
            show_legend: Whether to show legend.
            
        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Configure axes
        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        
        # Draw environment border
        ax.plot(
            [0, self.x_max, self.x_max, 0, 0],
            [0, 0, self.y_max, self.y_max, 0],
            'k-', linewidth=2, alpha=0.5
        )
        
        # Draw obstacles
        for i, (xo, yo, lx, ly) in enumerate(self.obstacles):
            rect = patches.Rectangle(
                (xo, yo), lx, ly,
                linewidth=1,
                edgecolor='black',
                facecolor='black',
                alpha=0.7,
                label='Obstacles' if i == 0 else None
            )
            ax.add_patch(rect)
        
        # Draw start and goal
        ax.plot(
            self.start[0], self.start[1],
            'go', markersize=10, label='Start'
        )
        ax.plot( 
            self.goal[0], self.goal[1],
            'rx', markersize=12, label='Goal'
        )
        
        # Draw path if provided
        if path is not None and len(path) > 0:
            path_arr = np.array(path)
            ax.plot(
                path_arr[:, 0], path_arr[:, 1],
                '-', color=path_color, linewidth=2, label=path_label
            )
        
        if show_legend:
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return fig, ax
