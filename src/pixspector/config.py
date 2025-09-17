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
        base = _read_yaml(defaults_path)
        src = defaults_path
        if override_path:
            over = _read_yaml(override_path)
            base = _deep_update(base, over)
            src = override_path
        return cls(data=base, source_path=src)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch nested values with dotted keys e.g. 'modules.ela.recompress_quality'."""
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
