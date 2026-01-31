from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.pretty import Pretty

class Console(RichConsole):
    def display(self, content, title="", style="none", border_style="none",):
        if not isinstance(content, str):
            content = Pretty(content)

        self.print(Panel(content, title=title, style=style, border_style=border_style, highlight=True, expand=True))

    def display_error(self, msg, title="Error"):
        self.display(msg, title=title, style="bold red", border_style="red")

    def print_error(self, msg):
        self.print(f"[bold red]Error:[/] {msg}")

