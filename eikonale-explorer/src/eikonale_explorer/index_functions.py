"""Refractive index functions for Eikonal equation.

This module provides various index functions N(x,y) that define
the propagation speed in the medium. Higher index = slower propagation.

Available index functions:
- uniform: Constant index everywhere
- diopter: Two regions with different indices (sea/beach analogy)
- gaussian: Localized high-index region (Gaussian bump)
- obstacle: Rectangular obstacles from scenario files
- island: Complex terrain with forest, mountains, river, sea
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class IndexConfig:
    """Configuration for index function generation."""
    nx: int
    ny: int
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)  # xmin, xmax, ymin, ymax


# =============================================================================
# Uniform Index
# =============================================================================

def create_uniform_index(config: IndexConfig, value: float = 1.0) -> np.ndarray:
    """Create uniform index field.
    
    Args:
        config: Grid configuration.
        value: Constant index value.
        
    Returns:
        2D array of shape (ny, nx) with constant value.
    """
    return np.full((config.ny, config.nx), value)


# =============================================================================
# Diopter Index (Two regions)
# =============================================================================

def create_diopter_index(
    config: IndexConfig,
    boundary: float = 0.5,
    index_low: float = 1.0,
    index_high: float = 10.0,
    axis: str = 'y'
) -> np.ndarray:
    """Create diopter index field (beach/sea analogy).
    
    Args:
        config: Grid configuration.
        boundary: Position of interface (0-1 normalized).
        index_low: Index in first region (fast).
        index_high: Index in second region (slow).
        axis: 'x' or 'y' for interface orientation.
        
    Returns:
        2D array with two distinct regions.
    """
    xmin, xmax, ymin, ymax = config.domain
    x = np.linspace(xmin, xmax, config.nx)
    y = np.linspace(ymin, ymax, config.ny)
    X, Y = np.meshgrid(x, y)
    
    if axis == 'y':
        return np.where(Y <= boundary, index_low, index_high)
    else:
        return np.where(X <= boundary, index_low, index_high)


# =============================================================================
# Gaussian Index (Localized bump)
# =============================================================================

def create_gaussian_index(
    config: IndexConfig,
    center: Tuple[float, float] = (0.6, 0.5),
    sigma_x: float = 50.0,
    sigma_y: float = 100.0,
    amplitude: float = 1.5,
    base_index: float = 1.0
) -> np.ndarray:
    """Create Gaussian bump index field.
    
    N(x,y) = base + amplitude * exp(-sigma_x*(x-cx)^2 - sigma_y*(y-cy)^2)
    
    Args:
        config: Grid configuration.
        center: (cx, cy) center of Gaussian.
        sigma_x, sigma_y: Spread parameters.
        amplitude: Height of bump.
        base_index: Background index.
        
    Returns:
        2D array with Gaussian bump.
    """
    xmin, xmax, ymin, ymax = config.domain
    x = np.linspace(xmin, xmax, config.nx)
    y = np.linspace(ymin, ymax, config.ny)
    X, Y = np.meshgrid(x, y)
    
    cx, cy = center
    return base_index + amplitude * np.exp(-sigma_x * (X - cx)**2 - sigma_y * (Y - cy)**2)


# =============================================================================
# Rectangular Obstacle Index
# =============================================================================

def create_obstacle_index(
    config: IndexConfig,
    obstacles: list,
    base_index: float = 1.0,
    obstacle_index: float = 1e6
) -> np.ndarray:
    """Create index field from rectangular obstacles.
    
    Args:
        config: Grid configuration.
        obstacles: List of (x, y, width, height) tuples.
        base_index: Index in free space.
        obstacle_index: Index inside obstacles (very high).
        
    Returns:
        2D array with obstacles marked.
    """
    xmin, xmax, ymin, ymax = config.domain
    x = np.linspace(xmin, xmax, config.nx)
    y = np.linspace(ymin, ymax, config.ny)
    X, Y = np.meshgrid(x, y)
    
    N = np.full((config.ny, config.nx), base_index)
    
    for (ox, oy, w, h) in obstacles:
        mask = (X >= ox) & (X <= ox + w) & (Y >= oy) & (Y <= oy + h)
        N[mask] = obstacle_index
        
    return N


# =============================================================================
# Island Index (Complex terrain)
# =============================================================================

def create_island_index(config: IndexConfig) -> np.ndarray:
    """Create island terrain index field.
    
    Features:
    - Beach (plage): index = 2
    - Forest (forêt): index = 1
    - Mountain (montagne): index = 10
    - River (rivière): index = 1e7
    - Sea (mer): index = 1e10
    - Bridge over river
    
    Returns:
        2D array with complex terrain.
    """
    # Terrain indices
    INDEX_BEACH = 2.0
    INDEX_FOREST = 1.0
    INDEX_MOUNTAIN = 10.0
    INDEX_RIVER = 1e7
    INDEX_SEA = 1e10
    
    # Island geometry
    center = np.array([0.5, 0.5])
    
    def point_index(x: float, y: float) -> float:
        # Distance from center
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Sea (outside island)
        if dist > 0.4:
            return INDEX_SEA
            
        # Beach (edge of island)
        if dist > 0.35:
            return INDEX_BEACH
            
        # River (vertical stripe)
        if 0.48 < x < 0.52:
            # Bridge
            if 0.45 < y < 0.55:
                return INDEX_FOREST
            return INDEX_RIVER
            
        # Mountain (center bump)
        mountain_dist = np.sqrt((x - 0.5)**2 + (y - 0.6)**2)
        if mountain_dist < 0.1:
            return INDEX_MOUNTAIN
            
        # Forest (default interior)
        return INDEX_FOREST
    
    xmin, xmax, ymin, ymax = config.domain
    x = np.linspace(xmin, xmax, config.nx)
    y = np.linspace(ymin, ymax, config.ny)
    
    N = np.array([[point_index(xi, yj) for xi in x] for yj in y])
    return N


# =============================================================================
# Custom Index from Function
# =============================================================================

def create_custom_index(
    config: IndexConfig,
    func: Callable[[float, float], float]
) -> np.ndarray:
    """Create index field from arbitrary function.
    
    Args:
        config: Grid configuration.
        func: Function f(x, y) -> index.
        
    Returns:
        2D array with custom index values.
    """
    xmin, xmax, ymin, ymax = config.domain
    x = np.linspace(xmin, xmax, config.nx)
    y = np.linspace(ymin, ymax, config.ny)
    
    return np.array([[func(xi, yj) for xi in x] for yj in y])


# =============================================================================
# Factory function
# =============================================================================

def get_index_function(name: str) -> Callable:
    """Get index function by name.
    
    Available: 'uniform', 'diopter', 'gaussian', 'obstacle', 'island', 'custom'
    """
    functions = {
        'uniform': create_uniform_index,
        'diopter': create_diopter_index,
        'gaussian': create_gaussian_index,
        'obstacle': create_obstacle_index,
        'island': create_island_index,
        'custom': create_custom_index,
    }
    
    if name not in functions:
        raise ValueError(f"Unknown index function: {name}. Available: {list(functions.keys())}")
        
    return functions[name]
