import typer

from typing import Optional

from .scripts.valid_input import main as valid_input
from .scripts.plot_environment import main as plot_environment

app = typer.Typer()

@app.command("valid-input")
def valid_input_cmd(
    file_path: Optional[str] = typer.Option(None, "-f", "--file-path")
):
    valid_input(file_path)
    
@app.command("plot-environment")
def plot_environment_cmd(
    file_path: Optional[str] = typer.Option(None, '-f', "--file-path")
):    
    plot_environment(file_path)

if __name__ == "__main__":
    app()
