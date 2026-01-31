from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """Runtime configuration loaded from YAML with optional overrides."""
    data: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @classmethod
    def load(cls, defaults_path: Path, override_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML files with validation.
        
        Args:
            defaults_path: Path to default config YAML
            override_path: Optional path to override config YAML
        
        Returns:
            Config object
        
        Raises:
            FileNotFoundError: If defaults_path doesn't exist
            ValueError: If YAML is invalid
        """
        if not defaults_path.exists():
            raise FileNotFoundError(f"Config file not found: {defaults_path}")
        
        try:
            base = _read_yaml(defaults_path)
        except Exception as e:
            raise ValueError(f"Failed to parse config {defaults_path}: {e}")
        
        src = defaults_path
        if override_path:
            if not override_path.exists():
                raise FileNotFoundError(f"Override config not found: {override_path}")
            try:
                over = _read_yaml(override_path)
                base = _deep_update(base, over)
                src = override_path
            except Exception as e:
                raise ValueError(f"Failed to parse override config {override_path}: {e}")
        
        return cls(data=base, source_path=src)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch nested values with dotted keys e.g. 'modules.ela.recompress_quality'.
        
        Args:
            dotted_key: Dot-separated key path (e.g., "modules.ela.quality")
            default: Default value if key not found
        
        Returns:
            Value at key path or default
        """
        if not isinstance(dotted_key, str) or not dotted_key:
            return default
        
        cur: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool = True
    max_file_size_mb: int = 50
    max_decode_pixels: int = 50_000_000
    max_memory_mb: int = 512
    max_cpu_seconds: int = 5
    worker_timeout_seconds: int = 10

    @classmethod
    def from_config(cls, cfg: "Config") -> "SandboxConfig":
        raw = cfg.get("sandbox", {}) or {}
        return cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            max_file_size_mb=int(raw.get("max_file_size_mb", cls.max_file_size_mb)),
            max_decode_pixels=int(raw.get("max_decode_pixels", cls.max_decode_pixels)),
            max_memory_mb=int(raw.get("max_memory_mb", cls.max_memory_mb)),
            max_cpu_seconds=int(raw.get("max_cpu_seconds", cls.max_cpu_seconds)),
            worker_timeout_seconds=int(raw.get("worker_timeout_seconds", cls.worker_timeout_seconds)),
        )


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict `base` with `over` (non-destructive)."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def find_defaults_path(start: Optional[Path] = None) -> Path:
    """Locate the repository defaults.yaml relative to ``start``.

    Walks up the directory tree from ``start`` (or this file) until a
    ``config/defaults.yaml`` file is found.
    """

    ref = start or Path(__file__).resolve()
    ref = ref if ref.is_dir() else ref.parent
    for parent in [ref, *ref.parents]:
        candidate = parent / "config" / "defaults.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config/defaults.yaml not found")
