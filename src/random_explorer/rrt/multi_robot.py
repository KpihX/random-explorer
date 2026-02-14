"""Multi-robot RRT* planner for 2-robot path planning problem."""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from ..environment import Environment
from ..utils import Console
from .planner import RRTPlanner
from .node import Node


class MultiRobotRRTPlanner:
    """RRT* algorithm to solve the two robots planning problem.
    
    Each robot consider the other one like an obstacle of radius R"""
        
    def __init__(
        self,
        env: Environment,
        max_iter: int = 1000,
        delta_s: float = 50.0,
        delta_r: float = 100.0,
        goal_bias: float = 0.05,
        goal_tolerance: float = 10.0
    ) -> None:
        """Initialize multi-robot planner."""
        self.env = env
        self.max_iter = max_iter
        self.delta_s = delta_s
        self.delta_r = delta_r
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance

        self.path1: Optional[List[Tuple[float,float]]] = None
        self.path2: Optional[List[Tuple[float,float]]] = None
        self.length1: float = float('inf')
        self.length2: float = float('inf')
    
    def _create_temporary_env(
        self,
        robot_id: int,
        other_robot_pos: Tuple[float, float]
    ) -> Environment:
        """Create environment with other robot as circular obstacle.
        
        Args:
            robot_id: 1 or 2 (which robot we're planning for).
            other_robot_pos: Current position of the other robot.
        
        Returns:
            Modified environment with circular obstacle.
        """
        temp_env = object.__new__(Environment)
        
        temp_env.x_max = self.env.x_max
        temp_env.y_max = self.env.y_max
        temp_env.radius = self.env.radius
        temp_env.start2 = self.env.start2
        temp_env.goal2 = self.env.goal2
        
        temp_env.obstacles = list(self.env.obstacles)
        
        other_x, other_y = other_robot_pos
        R = self.env.radius
        obs_x = other_x - R
        obs_y = other_y - R
        circular_obstacle = (obs_x, obs_y, 2 * R, 2 * R)
        temp_env.obstacles.append(circular_obstacle)
        
        if robot_id == 1:
            temp_env.start = self.env.start
            temp_env.goal = self.env.goal
        else:
            temp_env.start = self.env.start2
            temp_env.goal = self.env.goal2
        
        temp_env.console = Console()
        
        return temp_env
    
    def _check_robot_collision(
        self,
        path1: List[Tuple[float, float]],
        path2: List[Tuple[float, float]]
    ) -> bool:
        """Check if paths collide."""
        if not path1 or not path2:
            return False
        
        min_len = min(len(path1), len(path2))
        R = self.env.radius

        for i in range(min_len):
            p1 = np.array(path1[i])
            p2 = np.array(path2[i])
            dist = np.linalg.norm(p1-p2)

            if dist < 2 * R:
                return True

        return False
    
    def solve_alternating(
        self,
        intelligent_sampling: bool = False,
        optimized: bool = True,
        max_alternations: int = 5,
        show_progress: bool = True
    ) -> Tuple[
        Optional[List[Tuple[float, float]]],
        Optional[List[Tuple[float, float]]],
        float,
        float,
        int,
        int 
    ]:
        """Solve multi-robot problem with alternating planning."""
        current_pos1 = self.env.start
        current_pos2 = self.env.start2

        total_iterations1 = 0
        total_iterations2 = 0

        desc_parts = ["Multi-Robot RRT*"]
        if intelligent_sampling:
            desc_parts.append("intelligent")
        if optimized:
            desc_parts.append("optimized")
        desc = " + ".join(desc_parts)

        iterator = tqdm(
            range(max_alternations),
            desc=desc,
            disable=not show_progress,
            unit="alt",
            position=0,
            leave=True
        )

        for alternation in iterator:
            # Désactiver les barres internes pour éviter les conflits
            env1 = self._create_temporary_env(robot_id=1, other_robot_pos=current_pos2)
            planner1 = RRTPlanner(
                env=env1,
                max_iter=self.max_iter,
                delta_s=self.delta_s,
                delta_r=self.delta_r,
                goal_bias=self.goal_bias,
                goal_tolerance=self.goal_tolerance 
            )

            self.path1, self.length1, iter1 = planner1.solve(
                optimized=optimized,
                intelligent_sampling=intelligent_sampling,
                show_progress=False
            )

            total_iterations1 += iter1

            if self.path1 is None:
                iterator.set_postfix_str("Robot 1 failed")
                return None, None, float('inf'), float('inf'), total_iterations1, total_iterations2
            
            env2 = self._create_temporary_env(robot_id=2, other_robot_pos=current_pos1)
            planner2 = RRTPlanner(
                env=env2,
                max_iter=self.max_iter,
                delta_r=self.delta_r,
                delta_s=self.delta_s,
                goal_bias=self.goal_bias,
                goal_tolerance=self.goal_tolerance  
            )

            self.path2, self.length2, iter2 = planner2.solve(
                optimized=optimized,
                intelligent_sampling=intelligent_sampling,
                show_progress=False
            )

            total_iterations2 += iter2

            if self.path2 is None:
                iterator.set_postfix_str("Robot 2 failed")
                return None, None, float('inf'), float('inf'), total_iterations1, total_iterations2
            
            collision = self._check_robot_collision(self.path1, self.path2)

            # Mettre à jour la barre avec des informations détaillées
            status = {
                "L1": f"{self.length1:.0f}",
                "L2": f"{self.length2:.0f}",
                "I1": iter1,
                "I2": iter2,
                "coll": "Y" if collision else "N"
            }
            iterator.set_postfix(status)

            if not collision:
                iterator.set_postfix_str(f"Success! L1={self.length1:.0f}, L2={self.length2:.0f}")
                return self.path1, self.path2, self.length1, self.length2, total_iterations1, total_iterations2
            
            mid_idx1 = len(self.path1) // 2
            mid_idx2 = len(self.path2) // 2
            current_pos1 = self.path1[mid_idx1]
            current_pos2 = self.path2[mid_idx2]
        
        iterator.set_postfix_str(f"Max alternations. L1={self.length1:.0f}, L2={self.length2:.0f}")
        return self.path1, self.path2, self.length1, self.length2, total_iterations1, total_iterations2 

    def plot_solution(
        self,
        title: str = "Multi-Robot Solution",
        show_safety_zones: bool = True,
        figsize: tuple = (10, 10)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot both robot paths with their start/goal positions.
        
        Args:
            title: Plot title.
            show_safety_zones: Whether to draw safety circles around robots.
            figsize: Figure size (width, height).
        
        Returns:
            Tuple of (figure, axes).
        """
        # Create new figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Configure axes
        ax.set_xlim(0, self.env.x_max)
        ax.set_ylim(0, self.env.y_max)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Draw environment border
        ax.plot(
            [0, self.env.x_max, self.env.x_max, 0, 0],
            [0, 0, self.env.y_max, self.env.y_max, 0],
            'k-', linewidth=2, alpha=0.5
        )
        
        # Draw obstacles
        for i, (xo, yo, lx, ly) in enumerate(self.env.obstacles):
            rect = patches.Rectangle(
                (xo, yo), lx, ly,
                linewidth=1,
                edgecolor='black',
                facecolor='black',
                alpha=0.7,
                label='Obstacles' if i == 0 else None
            )
            ax.add_patch(rect)
        
        # Draw Robot 1 start and goal
        ax.plot(
            self.env.start[0], self.env.start[1],
            'go', markersize=12, markeredgecolor='darkgreen', 
            markeredgewidth=2, label='Start 1'
        )
        ax.plot(
            self.env.goal[0], self.env.goal[1],
            'g^', markersize=12, markeredgecolor='darkgreen',
            markeredgewidth=2, label='Goal 1'
        )
        
        # Draw Robot 2 start and goal
        ax.plot(
            self.env.start2[0], self.env.start2[1],
            'bo', markersize=12, markeredgecolor='darkblue',
            markeredgewidth=2, label='Start 2'
        )
        ax.plot(
            self.env.goal2[0], self.env.goal2[1],
            'b^', markersize=12, markeredgecolor='darkblue',
            markeredgewidth=2, label='Goal 2'
        )
        
        # Draw Robot 1 path
        if self.path1:
            path1_arr = np.array(self.path1)
            ax.plot(
                path1_arr[:, 0], path1_arr[:, 1],
                'g-', linewidth=2.5, alpha=0.8,
                label=f'Robot 1 (L={self.length1:.1f})'
            )
            # Add arrow to show direction
            mid_idx = len(path1_arr) // 2
            if len(path1_arr) > 1:
                dx = path1_arr[mid_idx, 0] - path1_arr[mid_idx-1, 0]
                dy = path1_arr[mid_idx, 1] - path1_arr[mid_idx-1, 1]
                ax.arrow(
                    path1_arr[mid_idx-1, 0], path1_arr[mid_idx-1, 1],
                    dx, dy, head_width=15, head_length=20,
                    fc='green', ec='darkgreen', alpha=0.6
                )
        
        if self.path2:
            path2_arr = np.array(self.path2)
            ax.plot(
                path2_arr[:, 0], path2_arr[:, 1],
                'b-', linewidth=2.5, alpha=0.8,
                label=f'Robot 2 (L={self.length2:.1f})'
            )
            mid_idx = len(path2_arr) // 2
            if len(path2_arr) > 1:
                dx = path2_arr[mid_idx, 0] - path2_arr[mid_idx-1, 0]
                dy = path2_arr[mid_idx, 1] - path2_arr[mid_idx-1, 1]
                ax.arrow(
                    path2_arr[mid_idx-1, 0], path2_arr[mid_idx-1, 1],
                    dx, dy, head_width=15, head_length=20,
                    fc='blue', ec='darkblue', alpha=0.6
                )
        
        if show_safety_zones:
            R = self.env.radius
            
            circle1_start = plt.Circle(
                self.env.start, R,
                color='green', fill=False,
                linestyle='--', linewidth=1.5, alpha=0.5,
                label=f'Safety zone (R={R:.0f})'
            )
            circle2_start = plt.Circle(
                self.env.start2, R,
                color='blue', fill=False,
                linestyle='--', linewidth=1.5, alpha=0.5
            )
            ax.add_patch(circle1_start)
            ax.add_patch(circle2_start)
        
        # Legend
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        return fig, ax
