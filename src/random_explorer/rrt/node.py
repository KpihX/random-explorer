"""Node class for RRT tree structure."""

from typing import Optional, Tuple
import numpy as np


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
