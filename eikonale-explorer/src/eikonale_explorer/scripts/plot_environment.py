from ..environment import Environment
from ..utils import console
import typer

app = typer.Typer()

@app.command()
def main(file: str = typer.Option(..., help="Path to scenario file")):
    """Simple visualization of the environment setup."""
    console.display_panel("Plotting Environment", style="green")
    
    try:
        env = Environment(file)
        env.plot(show=True, title=f"Environment: {file}")
    except Exception as e:
        console.display_error(f"Failed to plot: {e}")

if __name__ == "__main__":
    app()
