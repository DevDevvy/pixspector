"""Branding and visual elements for pixspector CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

LOGO = r"""
    ____  ____  __ _____ ____  _____ ______________  ____
   / __ \/  _/ |/ / ___// __ \/ ___// ____/ ____/ /_/ __ \/ __ \
  / /_/ // / |   /\__ \/ /_/ /\__ \/ __/ / /   / __/ / / / /_/ /
 / ____// / /   |___/ / ____/___/ / /___/ /___/ /_/ /_/ / _, _/
/_/   /___//_/|_/____/_/    /____/_____/\____/\__/_____/_/ |_|

"""

TAGLINE = "Classical Image Forensics • No ML Required • Explainable Results"

COLORS = {
    "primary": "cyan",
    "secondary": "blue",
    "accent": "magenta",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "muted": "dim white",
}


def print_logo(console: Console, show_tagline: bool = True) -> None:
    """Print the PIXSPECTOR logo with gradient effect."""
    lines = LOGO.strip().split("\n")
    
    # Create gradient effect from cyan to blue
    gradient_colors = ["cyan", "bright_cyan", "blue", "bright_blue", "blue", "cyan"]
    
    for i, line in enumerate(lines):
        color = gradient_colors[min(i, len(gradient_colors) - 1)]
        console.print(line, style=f"bold {color}")
    
    if show_tagline:
        console.print()
        console.print(f"[dim]{TAGLINE}[/dim]", justify="center")


def print_banner(console: Console, title: str, subtitle: str = "") -> None:
    """Print a styled banner."""
    text = Text()
    text.append(title, style="bold cyan")
    if subtitle:
        text.append("\n")
        text.append(subtitle, style="dim")
    
    console.print(Panel(
        text,
        border_style="cyan",
        padding=(1, 2),
    ))


def print_section_header(console: Console, title: str, icon: str = "▶") -> None:
    """Print a section header."""
    console.print()
    console.print(f"[bold cyan]{icon} {title}[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")


def print_success(console: Console, message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(console: Console, message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(console: Console, message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(console: Console, message: str) -> None:
    """Print an info message."""
    console.print(f"[cyan]ℹ[/cyan] {message}")


def get_status_icon(status: str) -> str:
    """Get a styled status icon."""
    icons = {
        "success": "[green]✓[/green]",
        "error": "[red]✗[/red]",
        "warning": "[yellow]⚠[/yellow]",
        "info": "[cyan]ℹ[/cyan]",
        "running": "[cyan]◉[/cyan]",
        "pending": "[dim]○[/dim]",
    }
    return icons.get(status.lower(), "[dim]•[/dim]")


def get_suspicion_badge(score: int) -> str:
    """Get a color-coded suspicion badge."""
    if score >= 70:
        return f"[bold red on white] {score:3d} [/bold red on white]"
    elif score >= 40:
        return f"[bold yellow on black] {score:3d} [/bold yellow on black]"
    else:
        return f"[bold green on black] {score:3d} [/bold green on black]"


def get_bucket_badge(bucket: str) -> str:
    """Get a styled bucket badge."""
    badges = {
        "High": "[bold white on red] HIGH [/bold white on red]",
        "Medium": "[bold black on yellow] MEDIUM [/bold black on yellow]",
        "Low": "[bold white on green] LOW [/bold white on green]",
    }
    return badges.get(bucket, f"[dim]{bucket}[/dim]")
