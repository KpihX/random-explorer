"""Utility functions and classes for eikonale_explorer.

This module provides helper utilities used across the package,
including console output formatting.
"""

from typing import Any, Union
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.pretty import Pretty


class Console(RichConsole):
    """Enhanced console for formatted output.
    
    Extends Rich Console with convenience methods for displaying
    formatted panels and error messages.
    """
    
    def display(
        self,
        content: Union[str, Any],
        title: str = "",
        style: str = "none",
        border_style: str = "none"
    ) -> None:
        """Display content in a formatted panel.
        
        Args:
            content: Text or object to display.
            title: Panel title.
            style: Text style.
            border_style: Panel border style.
        """
        if not isinstance(content, str):
            content = Pretty(content)
        
        self.print(Panel(
            content,
            title=title,
            style=style,
            border_style=border_style,
            highlight=True,
            expand=True
        ))
    
    def display_error(self, msg: str, title: str = "Error") -> None:
        """Display an error message in a red panel.
        
        Args:
            msg: Error message text.
            title: Panel title (default: "Error").
        """
        self.display(
            msg,
            title=title,
            style="bold red",
            border_style="red"
        )
    
    def print_error(self, msg: str) -> None:
        """Print an inline error message.
        
        Args:
            msg: Error message text.
        """
        self.print(f"[bold red]Error:[/] {msg}")

    def success(self, msg: str) -> None:
        """Print a success message.
        
        Args:
            msg: Message text.
        """
        self.print(f"[bold green]✓[/] {msg}")

    def info(self, msg: str) -> None:
        """Print an info message.
        
        Args:
            msg: Message text.
        """
        self.print(f"[bold blue]ℹ[/] {msg}")
        
    def warning(self, msg: str) -> None:
        """Print a warning message.
        
        Args:
            msg: Message text.
        """
        self.print(f"[bold yellow]⚠[/] {msg}")

    def panel(self, content: Any, title: str = "", style: str = "blue") -> None:
        """Legacy alias for display (kept for compatibility with my previous code)."""
        self.display(content, title=title, border_style=style)

# Global instance for convenience
console = Console()
