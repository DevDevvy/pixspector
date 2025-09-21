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

from . import __version__
from .config import Config, find_defaults_path
from .pipeline import analyze_single_image

app = typer.Typer(add_completion=False, help="pixspector — classical image forensics (no ML).")
console = Console()


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"[bold]pixspector[/] v{__version__}")


@app.command()
def doctor() -> None:
    """
    Environment checks (basic).
    """
    table = Table(title="pixspector doctor")
    table.add_column("Check")
    table.add_column("Result")

    table.add_row("Python", sys.version.split()[0])
    try:
        import cv2  # noqa
        table.add_row("OpenCV", "OK")
    except Exception as e:  # pragma: no cover
        table.add_row("OpenCV", f"MISS ({e})")
    try:
        import numpy as np  # noqa
        table.add_row("NumPy", "OK")
    except Exception as e:  # pragma: no cover
        table.add_row("NumPy", f"MISS ({e})")

    console.print(table)
    console.print(Panel.fit("If something is missing: `pip install -e .` or `pip install -r requirements.txt`"))


@app.command()
def analyze(
    src: str = typer.Argument(..., help="Image path or shell glob (quote globs)."),
    report: Path = typer.Option(Path("out"), "--report", "-r", help="Output folder for artifacts."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config (overrides defaults)."),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF generation."),
) -> None:
    """
    Run the full classical forensic pipeline on one or many images.
    """
    _ensure_outdir(report)

    # Load configuration
    try:
        defaults_path = find_defaults_path(Path(__file__))
    except FileNotFoundError:
        console.print("[red]defaults.yaml not found. Did you create the repo via the bootstrap script?[/]")
        raise typer.Exit(code=1)
    cfg = Config.load(defaults_path=defaults_path, override_path=config)

    # Expand globs
    paths: List[str] = []
    if any(ch in src for ch in ["*", "?", "[", "]"]):
        paths = sorted(glob.glob(src))
    else:
        paths = [src]

    if not paths:
        console.print("[red]No files matched.[/]")
        raise typer.Exit(code=1)

    table = Table(title="pixspector — results")
    table.add_column("Image")
    table.add_column("Suspicion")
    table.add_column("Bucket")
    table.add_column("JSON")
    table.add_column("PDF")

    for p in paths:
        pth = Path(p)
        if not pth.exists() or not pth.is_file():
            console.print(f"[yellow]Skip (not a file):[/] {p}")
            continue

        try:
            res = analyze_single_image(pth, cfg=cfg, out_dir=report, want_pdf=(not no_pdf))
            json_name = f"{pth.stem}_report.json"
            pdf_name = f"{pth.stem}_report.pdf" if not no_pdf else "-"
            table.add_row(
                pth.name,
                str(res.get("suspicion_index", "n/a")),
                str(res.get("bucket_label", "")),
                json_name,
                pdf_name,
            )
        except Exception as e:
            console.print(f"[red]Error analyzing {pth.name}:[/] {e}")

    console.print(table)
    console.print(Panel.fit(f"Artifacts saved to: {report.resolve()}"))
