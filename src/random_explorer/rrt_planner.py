"""Rapidly-exploring Random Tree (RRT*) path planner.

This module implements the RRT* algorithm for path planning, which builds
a tree structure to explore the configuration space and find paths.
RRT* includes rewiring for path optimization.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

from .environment import Environment


class Node:
    """A node in the RRT tree.
    
    Attributes:
        x: X coordinate of the node.
        y: Y coordinate of the node.
        parent: Parent node in the tree (None for root).
        cost: Cumulative cost from root to this node.
    """
    
    def __init__(self, x: float, y: float) -> None:
        """Initialize a tree node.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        self.x = x
        self.y = y
        self.parent: Optional['Node'] = None
        self.cost: float = 0.0
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get node position as tuple."""
        return (self.x, self.y)
    
    def distance_to(self, other: 'Node') -> float:
        """Compute Euclidean distance to another node."""
        return np.linalg.norm([self.x - other.x, self.y - other.y])
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Compute distance to a point (x, y)."""
        return np.linalg.norm([self.x - point[0], self.y - point[1]])


class RRTPlanner:
    """RRT* path planner for 2D environments.
    
    Builds a tree by randomly sampling points and extending the tree
    towards them. Includes neighbor search and rewiring for optimization.
    
    Attributes:
        env: The environment to plan in.
        max_iter: Maximum number of iterations.
        delta_s: Maximum extension step size.
        delta_r: Neighbor search radius for rewiring.
        goal_bias: Probability of sampling the goal directly.
        goal_tolerance: Distance threshold to consider goal reached.
        nodes: List of all nodes in the tree.
    """
    
    def __init__(
        self,
        env: Environment,
        max_iter: int = 1000,
        delta_s: float = 50.0,
        delta_r: float = 100.0,
        goal_bias: float = 0.05,
        goal_tolerance: float = 10.0
    ) -> None:
        """Initialize the RRT planner.
        
        Args:
            env: Environment to plan in.
            max_iter: Maximum number of iterations.
            delta_s: Maximum step size for tree extension.
            delta_r: Radius for neighbor search during rewiring.
            goal_bias: Probability of sampling goal directly (0-1).
            goal_tolerance: Distance to goal to consider success.
        """
        self.env = env
        self.max_iter = max_iter
        self.delta_s = delta_s
        self.delta_r = delta_r
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.nodes: List[Node] = []
        
        # Final results (set after solve())
        self.final_path: Optional[List[Tuple[float, float]]] = None
        self.path_length: float = float('inf')
    
    def _get_nearest_node(self, point: Tuple[float, float]) -> Node:
        """Find the node in the tree nearest to a point.
        
        Args:
            point: Target point (x, y).
            
        Returns:
            Nearest node in the tree.
        """
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            d = node.distance_to_point(point)
            if d < min_dist:
                min_dist = d
                nearest = node
        
        return nearest
    
    def _steer(
        self,
        from_node: Node,
        to_point: Tuple[float, float]
    ) -> Node:
        """Create a new node in the direction of a target point.
        
        If the target is within delta_s, reaches it exactly.
        Otherwise, creates a node at distance delta_s in that direction.
        
        Args:
            from_node: Starting node.
            to_point: Target point (x, y).
            
        Returns:
            New node at the steered position.
        """
        dist = from_node.distance_to_point(to_point)
        
        if dist <= self.delta_s:
            return Node(to_point[0], to_point[1])
        
        # Move delta_s in the direction of target
        theta = np.arctan2(
            to_point[1] - from_node.y,
            to_point[0] - from_node.x
        )
        new_x = from_node.x + self.delta_s * np.cos(theta)
        new_y = from_node.y + self.delta_s * np.sin(theta)
        
        return Node(new_x, new_y)
    
    def _check_collision(
        self,
        node1: Node,
        node2: Node,
        soft_mode: bool = True
    ) -> bool:
        """Check if segment between two nodes collides with obstacles.
        
        Uses Environment's evaluate_path_collision method.
        
        Args:
            node1: Start node.
            node2: End node.
            soft_mode: If True, use soft collision mode (default).
            
        Returns:
            True if collision detected, False otherwise.
        """
        segment = np.array([[node1.x, node1.y], [node2.x, node2.y]])
        collision_metric = self.env.evaluate_path_collision(segment, soft_mode)
        return collision_metric > 0
    
    def _get_neighbors(self, node: Node) -> List[Node]:
        """Find all nodes within delta_r of a given node.
        
        Args:
            node: Center node for neighbor search.
            
        Returns:
            List of neighboring nodes.
        """
        return [n for n in self.nodes if node.distance_to(n) <= self.delta_r]
    
    def _choose_best_parent(
        self,
        new_node: Node,
        nearest: Node,
        neighbors: List[Node]
    ) -> Node:
        """Find the best parent for a new node among neighbors.
        
        Chooses the neighbor that results in minimum cost to new_node.
        
        Args:
            new_node: Node being added.
            nearest: Default parent (nearest node).
            neighbors: Candidate parents within delta_r.
            
        Returns:
            Best parent node.
        """
        best_parent = nearest
        min_cost = nearest.cost + nearest.distance_to(new_node)
        
        for neighbor in neighbors:
            if self._check_collision(neighbor, new_node):
                continue
            
            cost = neighbor.cost + neighbor.distance_to(new_node)
            if cost < min_cost:
                min_cost = cost
                best_parent = neighbor
        
        return best_parent
    
    def _rewire(self, new_node: Node, neighbors: List[Node]) -> None:
        """Rewire neighbors if going through new_node is shorter.
        
        For each neighbor, check if routing through new_node reduces
        the path cost. If so, update the neighbor's parent.
        
        Args:
            new_node: Newly added node.
            neighbors: Nodes within delta_r of new_node.
        """
        for neighbor in neighbors:
            if neighbor == new_node.parent:
                continue
            
            new_cost = new_node.cost + new_node.distance_to(neighbor)
            
            if new_cost < neighbor.cost:
                if not self._check_collision(new_node, neighbor):
                    neighbor.parent = new_node
                    neighbor.cost = new_cost
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[float, float]]:
        """Reconstruct path from a node back to root.
        
        Follows parent pointers from the given node to the root,
        then reverses to get start-to-node order.
        
        Args:
            node: End node of the path.
            
        Returns:
            List of (x, y) points from start to node.
        """
        path = []
        current = node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]
    
    def _sample_random_point(self) -> Tuple[float, float]:
        """Sample a random point, with goal bias.
        
        With probability goal_bias, returns the goal position.
        Otherwise returns a uniform random point in the environment.
        
        Returns:
            Sampled point (x, y).
        """
        if np.random.rand() < self.goal_bias:
            return self.env.goal
        
        x = np.random.uniform(0, self.env.x_max)
        y = np.random.uniform(0, self.env.y_max)
        return (x, y)
    
    
    def _optimize_path(self, path):
    
        """Optimize the path using the triangular inequality.

        We consider the path and we try to found a direct path,
        from a node to another one.

        Args:
            path: Initial path 

        Returns:
            An optimized list of (x,y) points from start to end. 
        """
        if not path or len(path) < 3:
            return path
        
        optimized = [path[0]]  
        current_idx = 0
        
        while current_idx < len(path) - 1:

            farthest_valid = current_idx + 1
            
            for target_idx in range(len(path) - 1, current_idx, -1):

                node1 = Node(path[current_idx][0], path[current_idx][1])
                node2 = Node(path[target_idx][0], path[target_idx][1])
                
                if not self._check_collision(node1, node2):
                    farthest_valid = target_idx
                    break
            
            optimized.append(path[farthest_valid])
            current_idx = farthest_valid
        
        return optimized

    def _sample_intelligent(
        self,
        vertex_weight: float = 0.5,
        vertex_radius: float = 30.0,
        boundary_offset: float = 10.0
    ) -> Tuple[float, float]:
        """Intelligent sampling around vertices and boundary 
        of the obstacles.
        
        Randomly chooses between vertex sampling and boundary sampling
        based on weights, providing a natural mix of both strategies.
        
        Args:
            vertex_weight: Probability of choosing vertex over boundary (0-1).
            vertex_radius: Radius around vertices for sampling.
            boundary_offset: distance from obstacle edges.
        
        Returns:
            Sampled point (x, y).
        """
        
        obstacle = random.choice(self.env.obstacles)
        
        obs_x, obs_y, obs_lx, obs_ly = obstacle
        
        if random.random() < vertex_weight:

            vertices = [
                (obs_x, obs_y),
                (obs_x + obs_lx, obs_y),
                (obs_x, obs_y + obs_ly),
                (obs_x + obs_lx, obs_y + obs_ly)
            ]
            
            vertex = random.choice(vertices)
            
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, vertex_radius)
            
            x = vertex[0] + dist * np.cos(angle)
            y = vertex[1] + dist * np.sin(angle)
        
        else:

            side = random.randint(0, 3)
            
            if side == 0:  
                x = np.random.uniform(obs_x, obs_x + obs_lx)
                y = obs_y - boundary_offset
            elif side == 1: 
                x = obs_x + obs_lx + boundary_offset
                y = np.random.uniform(obs_y, obs_y + obs_ly)
            elif side == 2:  
                x = np.random.uniform(obs_x, obs_x + obs_lx)
                y = obs_y + obs_ly + boundary_offset
            else:  
                x = obs_x - boundary_offset
                y = np.random.uniform(obs_y, obs_y + obs_ly)
        
        x = np.clip(x, 0, self.env.x_max)
        y = np.clip(y, 0, self.env.y_max)
        
        return (x, y)
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total Euclidean length of a path.
        
        Args:
            path: List of (x, y) points representing the path.
            
        Returns:
            Total path length, or 0.0 if path has less than 2 points.
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total_length += np.sqrt(dx**2 + dy**2)
        
        return total_length

    def solve(
        self,
        optimized: bool = False,
        intelligent_sampling: bool = False,
        show_progress: bool = True
        ) -> Tuple[Optional[List[Tuple[float, float]]], float, int]:
        """Run RRT* algorithm to find a path.
        
        Args:
            optimized: Whether to apply post-processing path optimization.
            intelligent_sampling: Whether to use intelligent obstacle-based sampling.
            show_progress: Whether to show tqdm progress bar.
        
        Returns:
            Tuple of (final_path, path_length, iterations_used) where:
                - final_path: List of (x, y) points or None if not found
                - path_length: Path length or inf if not found
                - iterations_used: Number of iterations executed
        """
        start_node = Node(self.env.start[0], self.env.start[1])
        self.nodes = [start_node]
        
        goal_reached = False
        best_goal_node: Optional[Node] = None
        iterations_used = 0
        
        desc_parts = ["RRT*"]
        if intelligent_sampling:
            desc_parts.append("intelligent")
        if optimized:
            desc_parts.append("optimized")
        desc = " ".join(desc_parts) if len(desc_parts) > 1 else desc_parts[0]
        iterator = tqdm(range(self.max_iter), desc=desc, disable=not show_progress)
        
        for iteration in iterator:
            iterations_used = iteration + 1
            
            # Sample using the appropriate strategy
            if intelligent_sampling:
                rand_point = self._sample_intelligent()
            else:
                rand_point = self._sample_random_point()
            
            nearest = self._get_nearest_node(rand_point)
            
            new_node = self._steer(nearest, rand_point)
            
            if self._check_collision(nearest, new_node):
                continue
            
            neighbors = self._get_neighbors(new_node)
            
            best_parent = self._choose_best_parent(new_node, nearest, neighbors)
            
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + best_parent.distance_to(new_node)
            
            self.nodes.append(new_node)
            
            self._rewire(new_node, neighbors)
            
            dist_to_goal = new_node.distance_to_point(self.env.goal)
            if dist_to_goal <= self.goal_tolerance:
                # Try to connect directly to goal
                goal_node = Node(self.env.goal[0], self.env.goal[1])
                if not self._check_collision(new_node, goal_node):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist_to_goal
                    
                    if best_goal_node is None or goal_node.cost < best_goal_node.cost:
                        best_goal_node = goal_node
                        goal_reached = True
                        
                        if show_progress:
                            iterator.set_postfix({"best_cost": f"{goal_node.cost:.1f}"})
        
        if goal_reached and best_goal_node is not None:
            self.final_path = self._reconstruct_path(best_goal_node)
            if optimized:
                self.final_path = self._optimize_path(self.final_path)
                self.path_length = self._calculate_path_length(self.final_path)
            else:
                self.path_length = best_goal_node.cost
            return self.final_path, self.path_length, iterations_used
        
        # No path found - return closest approach
        closest = self._get_nearest_node(self.env.goal)
        dist = closest.distance_to_point(self.env.goal)
        
        if dist <= self.delta_s + self.goal_tolerance:
            self.final_path = self._reconstruct_path(closest)
            self.final_path.append(self.env.goal)
            if optimized:
                self.final_path = self._optimize_path(self.final_path)
                self.path_length = self._calculate_path_length(self.final_path)
            else:
                self.path_length = closest.cost + dist
            return self.final_path, self.path_length, iterations_used
        
        # No path found at all
        self.final_path = None
        self.path_length = float('inf')
        return None, float('inf'), iterations_used
    
    def plot_tree(
        self,
        path: Optional[List[Tuple[float, float]]] = None,
        title: Optional[str] = None,
        show_tree: bool = True,
        tree_alpha: float = 0.2,
        tree_linewidth: float = 0.5,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the RRT tree and optionally a path.
        
        Uses env.plot_environment as base and adds tree visualization.
        
        Args:
            path: Optional path to highlight (list of (x, y) points).
            title: Plot title. If None, auto-generated.
            show_tree: Whether to show all tree edges.
            tree_alpha: Transparency for tree edges.
            tree_linewidth: Line width for tree edges.
            ax: Optional axes to plot on.
            
        Returns:
            Tuple of (figure, axes) matplotlib objects.
        """
        # Use environment's plot_environment as base
        path_array = np.array(path) if path else None
        if title is None:
            title = f'RRT* - {len(self.nodes)} nodes'
        
        fig, ax = self.env.plot_environment(
            path=path_array,
            title=title,
            ax=ax,
            path_color='red',
            path_label='Solution',
            show_legend=False  # We'll add legend after tree
        )
        
        # Draw tree edges
        if show_tree:
            for node in self.nodes:
                if node.parent:
                    ax.plot(
                        [node.x, node.parent.x],
                        [node.y, node.parent.y],
                        'c-', alpha=tree_alpha, linewidth=tree_linewidth
                    )
        
        ax.legend()
        
        return fig, ax
