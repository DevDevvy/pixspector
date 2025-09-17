from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class C2PAResult:
    found: bool
    valid: bool
    tool: Optional[str]
    raw_json: Optional[Dict[str, Any]]
    error: Optional[str]


def has_c2patool() -> bool:
    """Check if c2patool is available on PATH."""
    return shutil.which("c2patool") is not None


def verify(path: Path, timeout: int = 10) -> C2PAResult:
    """
    Attempt to verify a C2PA manifest via the official `c2patool`.
    Gracefully degrades if the tool is missing or the file has no manifest.
    """
    exe = shutil.which("c2patool")
    if not exe:
        return C2PAResult(found=False, valid=False, tool=None, raw_json=None, error="c2patool not found")

    try:
        # Example: c2patool image.jpg -m
        proc = subprocess.run(
            [exe, str(path), "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip() or None

        # c2patool often prints JSON on stdout; try to parse
        raw = None
        valid = False
        if out:
            try:
                raw = json.loads(out)
                # Heuristic: status fields vary by version; attempt common keys
                valid = bool(raw.get("active_manifest") or raw.get("manifests"))
            except Exception:
                raw = {"raw_text": out}

        # If nothing obvious in stdout, still return presence of the tool
        return C2PAResult(
            found=True,
            valid=valid,
            tool=exe,
            raw_json=raw,
            error=err,
        )
    except subprocess.TimeoutExpired:
        return C2PAResult(found=True, valid=False, tool=exe, raw_json=None, error="timeout")
    except Exception as e:
        return C2PAResult(found=True, valid=False, tool=exe, raw_json=None, error=str(e))
