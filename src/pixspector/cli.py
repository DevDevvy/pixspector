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
    Environment checks: Python, key packages, optional external tools.
    (This is a stub; future batches will add c2patool checks, etc.)
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
    Run the classical forensic pipeline on one or many images.
    (Batch 1: pipeline is stubbed; later batches will fill in modules.)
    """
    _ensure_outdir(report)

    # Expand globs
    paths: List[str] = []
    if any(ch in src for ch in ["*", "?", "[", "]"]):
        paths = sorted(glob.glob(src))
    else:
        paths = [src]

    if not paths:
        console.print("[red]No files matched.[/]")
        raise typer.Exit(code=1)

    results_summary = []

    for p in paths:
        pth = Path(p)
        if not pth.exists() or not pth.is_file():
            console.print(f"[yellow]Skip (not a file):[/] {p}")
            continue

        # --- Placeholder: real pipeline will populate this dict in later batches ---
        result = {
            "input": str(pth),
            "status": "ok",
            "modules": [],
            "suspicion_index": None,  # filled when rules are implemented
            "notes": "Analysis stubs; modules will be added in upcoming batches.",
        }
        # Save minimal JSON now so the CLI feels complete
        out_json = report / f"{pth.stem}_report.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        results_summary.append((pth.name, result["status"], out_json.name))

    # Pretty print a simple summary table
    table = Table(title="pixspector — summary")
    table.add_column("Image")
    table.add_column("Status")
    table.add_column("Report JSON")
    for name, status, json_name in results_summary:
        table.add_row(name, status, json_name)

    console.print(table)

    if no_pdf:
        console.print(Panel.fit("PDF output skipped (--no-pdf)."))
    else:
        console.print(Panel.fit("PDF generation will be added in the Reporting batch."))

    console.print(Panel.fit(f"Artifacts saved to: {report.resolve()}"))


if __name__ == "__main__":
    app()
