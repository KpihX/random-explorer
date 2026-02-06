import typer
from .solve import main as solve_main
from .plot_environment import main as plot_main

app = typer.Typer(
    help="Eikonal Explorer CLI - Eikonal Equation Solver for Path Planning",
    add_completion=False,
    no_args_is_help=True
)

@app.command(name="solve")
def solve_cmd(
    file: str = typer.Option(..., help="Path to scenario file"),
    grid_size: int = typer.Option(128, help="Grid resolution (Nx=Ny)"),
    solver_iter: int = typer.Option(5000, help="Max iterations for Lax-Friedrichs"),
    output: str = typer.Option(None, help="Output image path")
):
    """Solve Eikonal equation and find path."""
    solve_main(file, grid_size, solver_iter, output)

@app.command(name="plot")
def plot_cmd(
    file: str = typer.Option(..., help="Path to scenario file")
):
    """Visualize simulation environment."""
    plot_main(file)

if __name__ == "__main__":
    app()
