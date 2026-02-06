from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table
from rich import box
from typing import Any, List, Optional

class Console:
    """Wrapper around rich.console.Console for standardized output."""
    
    def __init__(self):
        self._console = RichConsole()
        
    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)
        
    def panel(self, content: Any, title: str = "", style: str = "blue"):
        self._console.print(Panel(str(content), title=title, style=style, expand=False))
        
    def error(self, message: str):
        self._console.print(Panel(message, title="Error", style="bold red"))
        
    def success(self, message: str):
        self._console.print(f"[bold green]✓[/bold green] {message}")
        
    def info(self, message: str):
        self._console.print(f"[blue]ℹ[/blue] {message}")
        
    def warning(self, message: str):
        self._console.print(f"[yellow]⚠[/yellow] {message}")

    def table(self, title: str, columns: List[str], rows: List[List[Any]]):
        table = Table(title=title, box=box.ROUNDED)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(r) for r in row])
        self._console.print(table)

# Global console instance
console = Console()
