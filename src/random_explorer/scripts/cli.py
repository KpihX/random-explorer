"""Command-line interface for Random Explorer.

Provides commands for validating input, plotting environments,
and running hyperparameter grid search.
"""

import typer

from typing import Optional

from .valid_input import main as valid_input
from .plot_environment import main as plot_environment
from .grid_search import main as grid_search

app = typer.Typer()


@app.command("valid-input")
def valid_input_cmd(
    file_path: Optional[str] = typer.Option(None, "-f", "--file-path")
):
    """Validate a scenario input file."""
    valid_input(file_path)


@app.command("plot-environment")
def plot_environment_cmd(
    file_path: Optional[str] = typer.Option(None, '-f', "--file-path")
):
    """Plot a scenario environment."""
    plot_environment(file_path)


@app.command("grid-search")
def grid_search_cmd(
    scenarios: Optional[str] = typer.Option("4", "-s", "--scenarios", help="Scenario IDs (comma-separated, e.g., '3,4')"),
    runs: int = typer.Option(3, "-r", "--runs", help="Runs per configuration"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output JSON file path"),
    workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Number of parallel workers (default: auto)"),
    variant: str = typer.Option("basic", "-v", "--variant", help="PSO variant: basic, restart, sa, dl, adaptive")
):
    """Run grid search for PSO hyperparameters."""
    scenario_list = [int(s.strip()) for s in scenarios.split(",")] if scenarios else [4]
    grid_search(
        scenarios=scenario_list, 
        runs=runs, 
        output=output, 
        workers=workers,
        variant=variant
    )


if __name__ == "__main__":
    app()
