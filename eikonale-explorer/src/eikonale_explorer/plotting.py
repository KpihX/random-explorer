"""Visualization functions for Eikonal equation results.

This module provides plotting utilities for:
- Contour lines of the potential field phi
- Gradient fields
- Path trajectories
- Index maps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List
import matplotlib.colors as colors


def plot_contour_lines(
    phi: np.ndarray,
    source: Tuple[float, float],
    ax: Optional[plt.Axes] = None,
    levels: int = 30,
    add_colorbar: bool = True,
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> plt.Axes:
    """Plot contour lines (isolines) of the potential field.
    
    These represent wavefronts or isochrones.
    
    Args:
        phi: 2D potential field array.
        source: (x, y) source position to mark.
        ax: Optional matplotlib axes.
        levels: Number of contour levels.
        add_colorbar: Whether to add colorbar.
        cmap: Colormap name.
        title: Plot title.
        
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ny, nx = phi.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Mask very high values (obstacles)
    phi_masked = np.ma.masked_where(phi > 1e5, phi)
    
    cs = ax.contour(X, Y, phi_masked, levels=levels, cmap=cmap)
    
    if add_colorbar:
        plt.colorbar(cs, ax=ax, label='$\\phi$ (temps)')
    
    # Mark source
    ax.plot(source[0], source[1], 'r*', markersize=15, label='Source')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Lignes de niveau de $\\phi$')
    
    ax.legend()
    
    return ax


def plot_index_map(
    N: np.ndarray,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    cmap: str = 'terrain',
    title: str = "Carte d'indice N(x,y)"
) -> plt.Axes:
    """Plot the refractive index map.
    
    Args:
        N: 2D index field array.
        ax: Optional matplotlib axes.
        log_scale: Use logarithmic color scale (useful for obstacles).
        cmap: Colormap name.
        title: Plot title.
        
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if log_scale:
        # Use log scale for better visualization of high values
        norm = colors.LogNorm(vmin=max(N.min(), 0.1), vmax=N.max())
        im = ax.imshow(N, origin='lower', extent=[0, 1, 0, 1], 
                       cmap=cmap, norm=norm, aspect='equal')
    else:
        im = ax.imshow(N, origin='lower', extent=[0, 1, 0, 1], 
                       cmap=cmap, aspect='equal')
    
    plt.colorbar(im, ax=ax, label='Indice N')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    return ax


def plot_gradient_field(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    step: int = 4,
    title: str = "Champ de gradient $\\nabla\\phi$"
) -> plt.Axes:
    """Plot gradient field as quiver plot.
    
    Args:
        grad_x: X-component of gradient.
        grad_y: Y-component of gradient.
        ax: Optional matplotlib axes.
        step: Subsample step for arrow density.
        title: Plot title.
        
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ny, nx = grad_x.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Subsample for visibility
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              grad_x[::step, ::step], grad_y[::step, ::step],
              alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    return ax


def plot_path(
    path: np.ndarray,
    ax: Optional[plt.Axes] = None,
    source: Optional[Tuple[float, float]] = None,
    goal: Optional[Tuple[float, float]] = None,
    color: str = 'blue',
    linewidth: float = 2.0,
    label: str = 'Chemin optimal',
    title: Optional[str] = None
) -> plt.Axes:
    """Plot a reconstructed path.
    
    Args:
        path: Array of (x, y) points.
        ax: Optional matplotlib axes.
        source: Source point to mark.
        goal: Goal point to mark.
        color: Path color.
        linewidth: Path line width.
        label: Path label for legend.
        title: Plot title.
        
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth, label=label)
    
    if source is not None:
        ax.plot(source[0], source[1], 'g*', markersize=12, label='Source')
        
    if goal is not None:
        ax.plot(goal[0], goal[1], 'rx', markersize=10, label='Destination')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_comparison(
    phi: np.ndarray,
    paths: dict,
    source: Tuple[float, float],
    goal: Tuple[float, float],
    N: Optional[np.ndarray] = None,
    title: str = "Comparaison des méthodes"
) -> plt.Figure:
    """Plot comparison of multiple path reconstruction methods.
    
    Args:
        phi: Potential field.
        paths: Dict mapping method name to path array.
        source: Source position.
        goal: Goal position.
        N: Optional index map for background.
        title: Figure title.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Index map or phi contours with paths
    ax1 = axes[0]
    ny, nx = phi.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    phi_masked = np.ma.masked_where(phi > 1e5, phi)
    ax1.contour(X, Y, phi_masked, levels=20, cmap='gray', alpha=0.4)
    
    colors_list = ['blue', 'red', 'green', 'orange', 'purple']
    for idx, (name, path) in enumerate(paths.items()):
        if len(path) > 0:
            ax1.plot(path[:, 0], path[:, 1], color=colors_list[idx % len(colors_list)],
                    linewidth=2, label=name)
    
    ax1.plot(source[0], source[1], 'g*', markersize=15, label='Source')
    ax1.plot(goal[0], goal[1], 'rx', markersize=12, label='Goal')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title('Chemins reconstruits')
    
    # Right: Path lengths comparison
    ax2 = axes[1]
    names = []
    lengths = []
    for name, path in paths.items():
        if len(path) > 1:
            # Compute path length
            diffs = np.diff(path, axis=0)
            length = np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))
            names.append(name)
            lengths.append(length)
    
    if names:
        bars = ax2.bar(names, lengths, color=colors_list[:len(names)])
        ax2.set_ylabel('Longueur du chemin')
        ax2.set_title('Comparaison des longueurs')
        
        # Add value labels
        for bar, length in zip(bars, lengths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{length:.3f}', ha='center', va='bottom')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_obstacles(
    obstacles: List[Tuple[float, float, float, float]],
    ax: Optional[plt.Axes] = None,
    facecolor: str = 'black',
    alpha: float = 0.6
) -> plt.Axes:
    """Plot rectangular obstacles.
    
    Args:
        obstacles: List of (x, y, width, height) tuples.
        ax: Optional matplotlib axes.
        facecolor: Obstacle fill color.
        alpha: Transparency.
        
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    for (x, y, w, h) in obstacles:
        rect = Rectangle((x, y), w, h, facecolor=facecolor, alpha=alpha)
        ax.add_patch(rect)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    return ax
