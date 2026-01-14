from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


@dataclass
class TimestampResult:
    provider: str
    token: Optional[bytes] = None
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimestampProvider(Protocol):
    def timestamp(self, data: bytes) -> TimestampResult:
        ...


class NullTimestampProvider:
    def timestamp(self, data: bytes) -> TimestampResult:
        return TimestampResult(provider="none")


def _hash_bytes(data: bytes) -> Dict[str, str]:
    hashes: Dict[str, str] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha3_256": hashlib.sha3_256(data).hexdigest(),
    }
    try:
        import blake3
    except ImportError:
        return hashes

    hashes["blake3"] = blake3.blake3(data).hexdigest()
    return hashes


def _write_timestamp_token(out_dir: Path, token: bytes) -> Path:
    token_dir = out_dir / "timestamp_tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token_path = token_dir / f"timestamp_{stamp}.tsr"
    token_path.write_bytes(token)
    return token_path


def _record_timestamp(
    data: bytes,
    out_dir: Path,
    provider: Optional[TimestampProvider],
) -> Dict[str, Any]:
    provider = provider or NullTimestampProvider()
    result = provider.timestamp(data)
    token_path = None
    token_b64 = None
    if result.token:
        token_path = _write_timestamp_token(out_dir, result.token)
        token_b64 = base64.b64encode(result.token).decode("utf-8")
    return {
        "provider": result.provider,
        "token_path": str(token_path) if token_path else None,
        "token_b64": token_b64,
        "reference": result.reference,
        "metadata": result.metadata,
    }


def _append_chain_of_custody(out_dir: Path, entry: Dict[str, Any]) -> Path:
    log_path = out_dir / "chain_of_custody.jsonl"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return log_path


def intake_file(
    path: Path,
    out_dir: Path,
    timestamp_provider: Optional[TimestampProvider] = None,
) -> Dict[str, Any]:
    data = path.read_bytes()
    hashes = _hash_bytes(data)
    timestamp_info = _record_timestamp(data, out_dir, timestamp_provider)
    entry = {
        "event": "intake",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_path": str(path),
        "hashes": hashes,
        "timestamp_info": timestamp_info,
    }
    log_path = _append_chain_of_custody(out_dir, entry)
    return {
        "hashes": hashes,
        "timestamp": timestamp_info,
        "chain_of_custody_log": str(log_path),
    }
