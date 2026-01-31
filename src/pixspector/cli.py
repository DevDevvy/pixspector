from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

from . import __version__
from .config import Config, find_defaults_path
from .pipeline import analyze_single_image
from .branding import (
    print_logo, print_banner, print_section_header,
    print_success, print_error, print_warning, print_info,
    get_status_icon, get_suspicion_badge, get_bucket_badge
)

app = typer.Typer(
    add_completion=False,
    help="Classical image forensics toolkit (no ML required)",
    rich_markup_mode="rich",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """PIXSPECTOR - Classical Image Forensics Toolkit"""
    if ctx.invoked_subcommand is None:
        print_logo(console)
        console.print()
        console.print(f"[bold]Version:[/bold] {__version__}", justify="center")
        console.print()
        console.print("[cyan]Available Commands:[/cyan]")
        console.print("  [bold]analyze[/bold]    - Analyze images for manipulation and AI generation")
        console.print("  [bold]summarize[/bold]  - Summarize multiple analysis reports")
        console.print("  [bold]doctor[/bold]     - Check system configuration and dependencies")
        console.print("  [bold]version[/bold]    - Show version information")
        console.print()
        console.print("[dim]Run [cyan]pixspector COMMAND --help[/cyan] for more information on a command.[/dim]")
        console.print()


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def version() -> None:
    """Show version and system information."""
    print_logo(console)
    console.print()
    console.print(Panel(
        f"[bold cyan]Version:[/bold cyan] {__version__}\n"
        f"[bold cyan]Python:[/bold cyan] {sys.version.split()[0]}\n"
        f"[bold cyan]Platform:[/bold cyan] {sys.platform}",
        title="[bold]System Information[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))
    console.print()
    console.print("[dim]Run [cyan]pixspector --help[/cyan] for usage information[/dim]")
    console.print("[dim]Run [cyan]pixspector doctor[/cyan] for detailed diagnostics[/dim]")


@app.command()
def doctor() -> None:
    """
    Check system configuration and dependencies.
    """
    print_logo(console, show_tagline=False)
    console.print()
    print_section_header(console, "System Diagnostics", "🔍")
    console.print()
    
    table = Table(
        title="[bold cyan]Environment Check[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True
    )
    table.add_column("Component", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    # Python version
    py_version = sys.version.split()[0]
    table.add_row("Python", f"[green]✓[/green] v{py_version}", "Required: >=3.10")
    
    # Core dependencies
    try:
        import cv2  # noqa
        cv2_version = getattr(cv2, "__version__", "unknown")
        table.add_row("OpenCV", "[green]✓ Installed[/]", f"v{cv2_version}")
    except Exception as e:  # pragma: no cover
        table.add_row("OpenCV", "[red]✗ Missing[/]", str(e)[:40])
    
    try:
        import numpy as np  # noqa
        table.add_row("NumPy", "[green]✓ Installed[/]", f"v{np.__version__}")
    except Exception as e:  # pragma: no cover
        table.add_row("NumPy", "[red]✗ Missing[/]", str(e)[:40])
    
    try:
        import PIL  # noqa
        table.add_row("Pillow", "[green]✓ Installed[/]", f"v{PIL.__version__}")
    except Exception as e:  # pragma: no cover
        table.add_row("Pillow", "[red]✗ Missing[/]", str(e)[:40])
    
    # Optional dependencies
    try:
        from . import core  # noqa
        from .core import c2pa
        if c2pa.has_c2patool():
            table.add_row("C2PA Tool", "[green]✓ Available[/]", "Provenance verification enabled")
        else:
            table.add_row("C2PA Tool", "[yellow]○ Optional[/]", "Install for C2PA support")
    except Exception:
        table.add_row("C2PA Tool", "[yellow]○ Optional[/]", "Install for C2PA support")
    
    # System info
    import platform
    table.add_row("Platform", f"[cyan]{platform.system()}[/]", f"{platform.machine()}")
    
    # CPU count
    cpu_count = os.cpu_count() or 1
    table.add_row("CPU Cores", f"[cyan]{cpu_count}[/]", "Parallel processing capacity")

    console.print(table)
    console.print()
    
    # Status summary
    console.print(Panel(
        "[bold green]✓ System is ready for forensic analysis![/bold green]\n\n"
        "[bold]Quick Start:[/bold]\n"
        "  [cyan]pixspector analyze image.jpg[/cyan]\n\n"
        "[bold]Installation Help:[/bold]\n"
        "  Dependencies: [cyan]pip install -e .[/cyan]\n"
        "  C2PA Support: [cyan]cargo install c2patool[/cyan] [dim](optional)[/dim]",
        title="[bold]Status[/bold]",
        border_style="green",
        box=box.ROUNDED
    ))


@app.command()
def analyze(
    src: str = typer.Argument(..., help="Image path or shell glob (quote globs)."),
    report: Path = typer.Option(Path("out"), "--report", "-r", help="Output folder for artifacts."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config (overrides defaults)."),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF generation."),
    max_size_mb: int = typer.Option(50, "--max-size", help="Max file size in MB to process (0 = unlimited)."),
    continue_on_error: bool = typer.Option(True, "--continue", help="Continue processing other images if one fails."),
) -> None:
    """
    Analyze images for manipulation, AI generation, and authenticity.
    """
    print_logo(console, show_tagline=False)
    console.print()
    print_section_header(console, "Forensic Analysis Pipeline", "🔍")
    console.print()
    # Validate output directory
    try:
        _ensure_outdir(report)
    except PermissionError:
        print_error(console, f"Permission denied creating output directory: {report}")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(console, f"Error creating output directory {report}: {e}")
        raise typer.Exit(code=1)

    # Validate max_size_mb
    if max_size_mb < 0:
        print_error(console, "--max-size must be non-negative (0 = unlimited)")
        raise typer.Exit(code=1)

    # Load configuration with error handling
    try:
        defaults_path = find_defaults_path(Path(__file__))
    except FileNotFoundError:
        print_error(console, "defaults.yaml not found. Did you create the repo via the bootstrap script?")
        raise typer.Exit(code=1)
    
    try:
        cfg = Config.load(defaults_path=defaults_path, override_path=config)
        print_success(console, f"Configuration loaded from {defaults_path.name}")
    except Exception as e:
        print_error(console, f"Error loading configuration: {e}")
        raise typer.Exit(code=1)

    # Expand globs with validation
    console.print("[dim]Scanning for images...[/dim]")
    paths: List[str] = []
    if any(ch in src for ch in ["*", "?", "[", "]"]):
        paths = sorted(glob.glob(src))
    else:
        paths = [src]

    if not paths:
        print_error(console, "No files matched the pattern")
        raise typer.Exit(code=1)

    # Validate and filter paths
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif", ".bmp"}
    validated_paths = []
    skipped = {"not_file": 0, "too_large": 0, "unsupported": 0, "not_found": 0}
    
    for p in paths:
        pth = Path(p)
        
        if not pth.exists():
            skipped["not_found"] += 1
            console.print(f"[dim]  ⊘ Skip: {p} (not found)[/dim]")
            continue
            
        if not pth.is_file():
            skipped["not_file"] += 1
            console.print(f"[dim]  ⊘ Skip: {p} (not a file)[/dim]")
            continue
        
        # Check file extension
        if pth.suffix.lower() not in valid_extensions:
            skipped["unsupported"] += 1
            console.print(f"[dim]  ⊘ Skip: {pth.name} (unsupported format)[/dim]")
            continue
        
        # Check file size
        if max_size_mb > 0:
            size_mb = pth.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                skipped["too_large"] += 1
                console.print(f"[dim]  ⊘ Skip: {pth.name} (too large: {size_mb:.1f} MB)[/dim]")
                continue
        
        validated_paths.append(pth)
    
    if not validated_paths:
        print_error(console, "No valid image files to process")
        if sum(skipped.values()) > 0:
            console.print(f"[yellow]Skipped: {skipped['not_found']} not found, {skipped['not_file']} not files, "
                        f"{skipped['too_large']} too large, {skipped['unsupported']} unsupported[/yellow]")
        raise typer.Exit(code=1)
    
    console.print()
    print_success(console, f"Found {len(validated_paths)} valid image(s) to analyze")
    if sum(skipped.values()) > 0:
        print_warning(console, f"Skipped {sum(skipped.values())} invalid files")
    console.print()

    table = Table(
        title=f"[bold cyan]Analysis Results[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=False
    )
    table.add_column("Image", style="bold white", no_wrap=True)
    table.add_column("Suspicion", justify="center", width=10)
    table.add_column("Bucket", justify="center", width=10)
    table.add_column("Reports", style="dim")
    table.add_column("Status", justify="center", width=8)

    success_count = 0
    error_count = 0
    
    for idx, pth in enumerate(validated_paths, 1):
        console.print(f"[cyan]▶[/cyan] [{idx}/{len(validated_paths)}] Analyzing [bold]{pth.name}[/bold]...")
        
        try:
            res = analyze_single_image(pth, cfg=cfg, out_dir=report, want_pdf=(not no_pdf))
            json_name = f"{pth.stem}_report.json"
            pdf_info = f"+ PDF" if not no_pdf else ""
            
            suspicion = res.get("suspicion_index", 0)
            bucket = res.get("bucket_label", "Unknown")
            
            table.add_row(
                pth.name,
                get_suspicion_badge(suspicion),
                get_bucket_badge(bucket),
                f"JSON {pdf_info}",
                "[green]✓[/green]"
            )
            success_count += 1
            print_success(console, f"Completed {pth.name}")
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Processing interrupted by user[/yellow]")
            break
        except Exception as e:
            error_count += 1
            error_msg = str(e)[:80]
            print_error(console, f"{pth.name}: {error_msg}")
            table.add_row(
                pth.name,
                "[dim]--[/dim]",
                "[dim]--[/dim]",
                "[red]Failed[/red]",
                "[red]✗[/red]"
            )
            if not continue_on_error:
                print_warning(console, "Stopping due to error (use --continue to process remaining files)")
                break
        console.print()

    console.print()
    console.print(table)
    console.print()
    
    # Final summary
    if error_count == 0:
        status_color = "green"
        status_icon = "✓"
        status_text = "All analyses completed successfully!"
    elif success_count > 0:
        status_color = "yellow"
        status_icon = "⚠"
        status_text = "Analysis completed with some errors"
    else:
        status_color = "red"
        status_icon = "✗"
        status_text = "Analysis failed"
    
    console.print(Panel(
        f"[bold {status_color}]{status_icon} {status_text}[/bold {status_color}]\n\n"
        f"[bold]Results:[/bold]\n"
        f"  [green]✓[/green] Successful: {success_count}\n"
        f"  [red]✗[/red] Failed: {error_count}\n\n"
        f"[bold]Output:[/bold]\n"
        f"  [cyan]{report.resolve()}[/cyan]",
        title="[bold]Summary[/bold]",
        border_style=status_color,
        box=box.ROUNDED
    ))
    
    if error_count > 0 and success_count > 0:
        raise typer.Exit(code=2)  # Partial success
    elif error_count > 0:
        raise typer.Exit(code=1)  # All failed


@app.command()
def summarize(
    report_dir: Path = typer.Argument(..., help="Directory containing *_report.json files."),
    sort_by: str = typer.Option("suspicion", "--sort", "-s", help="Sort by: suspicion, name, bucket"),
) -> None:
    """
    Generate a summary report from multiple analysis results.
    """
    print_logo(console, show_tagline=False)
    console.print()
    print_section_header(console, "Report Summary", "📊")
    console.print()
    
    if not report_dir.exists():
        print_error(console, f"Directory not found: {report_dir}")
        raise typer.Exit(code=1)
    
    if not report_dir.is_dir():
        print_error(console, f"Not a directory: {report_dir}")
        raise typer.Exit(code=1)
    
    # Find all JSON reports
    console.print("[dim]Scanning for report files...[/dim]")
    report_files = list(report_dir.glob("*_report.json"))
    
    if not report_files:
        print_warning(console, f"No report files found in {report_dir}")
        console.print("[dim]Report files should match pattern: *_report.json[/dim]")
        raise typer.Exit(code=0)
    
    print_success(console, f"Found {len(report_files)} report(s)")
    console.print()
    
    # Load and parse reports
    reports = []
    for rfile in report_files:
        try:
            with open(rfile, "r") as f:
                data = json.load(f)
                reports.append({
                    "file": rfile.stem.replace("_report", ""),
                    "suspicion": data.get("suspicion_index", 0),
                    "bucket": data.get("bucket_label", "Unknown"),
                    "format": data.get("metadata", {}).get("format", "?"),
                    "dimensions": f"{data.get('image', {}).get('width', 0)}x{data.get('image', {}).get('height', 0)}",
                    "evidence_count": len(data.get("evidence", [])),
                })
        except Exception as e:
            print_warning(console, f"Could not parse {rfile.name}: {str(e)[:60]}")
    
    if not reports:
        print_error(console, "No valid reports could be loaded")
        raise typer.Exit(code=1)
    
    # Sort reports
    if sort_by == "suspicion":
        reports.sort(key=lambda x: x["suspicion"], reverse=True)
    elif sort_by == "name":
        reports.sort(key=lambda x: x["file"])
    elif sort_by == "bucket":
        reports.sort(key=lambda x: (x["bucket"], -x["suspicion"]))
    
    # Display summary table
    table = Table(
        title=f"[bold cyan]Analysis Summary[/bold cyan] [dim]({len(reports)} images)[/dim]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=False
    )
    table.add_column("Image", style="bold white")
    table.add_column("Suspicion", justify="center", width=10)
    table.add_column("Bucket", justify="center", width=10)
    table.add_column("Format", justify="center", width=8)
    table.add_column("Size", justify="right", width=12)
    table.add_column("Evidence", justify="center", width=10)
    
    for rep in reports:
        table.add_row(
            rep["file"],
            get_suspicion_badge(rep["suspicion"]),
            get_bucket_badge(rep["bucket"]),
            f"[cyan]{rep['format']}[/cyan]",
            f"[dim]{rep['dimensions']}[/dim]",
            f"[yellow]{rep['evidence_count']}[/yellow]",
        )
    
    console.print(table)
    console.print()
    
    # Statistics
    avg_suspicion = sum(r["suspicion"] for r in reports) / len(reports)
    bucket_counts = {}
    for r in reports:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1
    
    # Create stats summary
    stats_text = f"[bold]Overall Statistics:[/bold]\n\n"
    stats_text += f"  Average Suspicion: [cyan]{avg_suspicion:.1f}[/cyan]\n\n"
    stats_text += f"[bold]Risk Distribution:[/bold]\n"
    
    for bucket in ["High", "Medium", "Low"]:
        count = bucket_counts.get(bucket, 0)
        pct = 100 * count / len(reports) if reports else 0
        if bucket == "High":
            color = "red"
        elif bucket == "Medium":
            color = "yellow"
        else:
            color = "green"
        bar_length = int(pct / 5)  # Scale to 20 char max
        bar = "█" * bar_length
        stats_text += f"  [{color}]{bucket:8}[/{color}] [{color}]{bar}[/{color}] {count:3d} [dim]({pct:5.1f}%)[/dim]\n"
    
    console.print(Panel(
        stats_text,
        title="[bold]Summary Statistics[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))
