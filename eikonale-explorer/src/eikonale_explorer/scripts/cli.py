"""Command-line interface for eikonale-explorer.

Available commands:
- solve: Solve Eikonal equation and find path
- plot: Visualize environment
"""

import typer
from typing import Optional
from .solve import main as solve_main
from .plot_environment import main as plot_main

app = typer.Typer(
    name="eikonal-explorer",
    help="Eikonal Equation Solver for Path Planning",
    add_completion=False,
    no_args_is_help=True
)


@app.command(name="solve")
def solve_cmd(
    file: str = typer.Option(..., "--file", "-f", help="Path to scenario file"),
    grid_size: int = typer.Option(128, "--grid-size", "-n", help="Grid resolution (Nx=Ny)"),
    max_iter: int = typer.Option(5000, "--max-iter", "-m", help="Max solver iterations"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output image path"),
):
    """Solve Eikonal equation and reconstruct optimal path."""
    solve_main(file, grid_size, max_iter, output)


@app.command(name="plot")
def plot_cmd(
    file: str = typer.Option(..., "--file", "-f", help="Path to scenario file"),
):
    """Visualize environment from scenario file."""
    plot_main(file=file)


if __name__ == "__main__":
    app()
