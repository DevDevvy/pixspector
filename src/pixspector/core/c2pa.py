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
    manifest_store_raw: Optional[bytes]
    manifest_store_error: Optional[str]
    claims: Optional[Dict[str, Any]]
    validation: Optional[Dict[str, Any]]
    error: Optional[str]


def has_c2patool() -> bool:
    """Check if c2patool is available on PATH."""
    return shutil.which("c2patool") is not None


def _parse_json_output(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_text": raw}


def _select_active_manifest(raw: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not raw:
        return None, None
    manifests = raw.get("manifests") if isinstance(raw, dict) else None
    if not isinstance(manifests, dict) or not manifests:
        return None, None
    active = raw.get("active_manifest")
    if active and active in manifests:
        return active, manifests.get(active)
    first_key = next(iter(manifests.keys()), None)
    return first_key, manifests.get(first_key) if first_key else None


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "valid" if value else "invalid"
    if isinstance(value, (int, float)):
        return "valid" if value else "invalid"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "valid", "passed", "pass", "ok", "success"}:
            return "valid"
        if lowered in {"false", "invalid", "failed", "fail", "error"}:
            return "invalid"
        if lowered in {"warning", "warn"}:
            return "warning"
        return lowered
    return str(value)


def _find_first_value(data: Any, keys: set[str]) -> Optional[Any]:
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            found = _find_first_value(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_first_value(item, keys)
            if found is not None:
                return found
    return None


def _extract_validation(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    manifest_label, manifest = _select_active_manifest(raw)
    validation_source = manifest or raw or {}
    signature_keys = {
        "signature",
        "signature_info",
        "signatureInfo",
        "signature_valid",
        "signatureValid",
        "signature_status",
        "signatureStatus",
    }
    cert_keys = {
        "certificate_chain",
        "certificateChain",
        "cert_chain",
        "certChain",
        "certificates",
        "certs",
        "certChainStatus",
        "certificateStatus",
    }
    timestamp_keys = {
        "timestamp",
        "time_stamp",
        "timeStamp",
        "rfc3161",
        "rfc3161_timestamp",
        "timestamp_token",
        "timeStampToken",
        "tsp",
    }

    signature_value = _find_first_value(validation_source, signature_keys)
    cert_value = _find_first_value(validation_source, cert_keys)
    timestamp_value = _find_first_value(validation_source, timestamp_keys)

    signature_status = _normalize_status(signature_value if not isinstance(signature_value, dict) else signature_value.get("status"))
    cert_status = _normalize_status(cert_value if not isinstance(cert_value, dict) else cert_value.get("status"))
    timestamp_status = _normalize_status(
        timestamp_value if not isinstance(timestamp_value, dict) else timestamp_value.get("status")
    )

    return {
        "active_manifest": manifest_label,
        "signature": {"status": signature_status or "unknown", "details": signature_value},
        "certificate_chain": {"status": cert_status or "unknown", "details": cert_value},
        "timestamp": {"status": timestamp_status or "unknown", "details": timestamp_value},
    }


def _normalize_assertions(manifest: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not isinstance(manifest, dict):
        return []
    assertions = manifest.get("assertions", [])
    normalized: list[Dict[str, Any]] = []
    if isinstance(assertions, dict):
        for label, data in assertions.items():
            normalized.append({"label": label, "data": data})
    elif isinstance(assertions, list):
        for entry in assertions:
            if isinstance(entry, dict):
                label = entry.get("label") or entry.get("name") or ""
                data = entry.get("data") if "data" in entry else entry.get("value", entry)
                normalized.append({"label": label, "data": data})
    return normalized


def _extract_claims(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    _, manifest = _select_active_manifest(raw)
    assertions = _normalize_assertions(manifest)
    actions: list[Any] = []
    software_agents: list[Any] = []
    ai_assertions: list[Any] = []

    for assertion in assertions:
        label = str(assertion.get("label", "")).lower()
        data = assertion.get("data")

        if "assertion/ai" in label or label.endswith(".ai") or "c2pa.ai" in label:
            ai_assertions.append(data)

        if "actions" in label:
            if isinstance(data, dict) and "actions" in data:
                actions.extend(data.get("actions") or [])
            elif data:
                actions.append(data)

        if "software" in label or "softwareagent" in label:
            if isinstance(data, dict):
                for key in ("softwareAgent", "software_agent", "softwareAgents"):
                    if key in data:
                        agent_value = data.get(key)
                        if isinstance(agent_value, list):
                            software_agents.extend(agent_value)
                        else:
                            software_agents.append(agent_value)
            elif data:
                software_agents.append(data)

        if isinstance(data, dict) and "softwareAgent" in data:
            software_agents.append(data.get("softwareAgent"))

    claim_generator = None
    claim_generator_info = None
    if isinstance(manifest, dict):
        claim_generator = manifest.get("claim_generator") or manifest.get("claimGenerator")
        claim_generator_info = manifest.get("claim_generator_info") or manifest.get("claimGeneratorInfo")

    return {
        "assertion_ai": ai_assertions,
        "softwareAgent": software_agents,
        "actions": actions,
        "claimGenerator": claim_generator,
        "claimGeneratorInfo": claim_generator_info,
    }


def _extract_manifest_store(exe: str, path: Path, timeout: int) -> tuple[Optional[bytes], Optional[str]]:
    try:
        proc = subprocess.run(
            [exe, str(path), "--manifest-store"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="ignore").strip() or f"exit {proc.returncode}"
            return None, err
        if not proc.stdout:
            return None, "empty manifest store"
        return proc.stdout, None
    except subprocess.TimeoutExpired:
        return None, "manifest store timeout"
    except Exception as exc:
        return None, str(exc)


def verify(path: Path, timeout: int = 10) -> C2PAResult:
    """
    Attempt to verify a C2PA manifest via the official `c2patool`.
    Gracefully degrades if the tool is missing or the file has no manifest.
    """
    exe = shutil.which("c2patool")
    if not exe:
        return C2PAResult(
            found=False,
            valid=False,
            tool=None,
            raw_json=None,
            manifest_store_raw=None,
            manifest_store_error=None,
            claims=None,
            validation=None,
            error="c2patool not found",
        )

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

        raw = _parse_json_output(out)
        valid = bool(raw and isinstance(raw, dict) and (raw.get("active_manifest") or raw.get("manifests")))

        manifest_store_raw, manifest_store_error = _extract_manifest_store(exe, path, timeout=timeout)
        claims = _extract_claims(raw)
        validation = _extract_validation(raw)

        # If nothing obvious in stdout, still return presence of the tool
        return C2PAResult(
            found=True,
            valid=valid,
            tool=exe,
            raw_json=raw,
            manifest_store_raw=manifest_store_raw,
            manifest_store_error=manifest_store_error,
            claims=claims,
            validation=validation,
            error=err,
        )
    except subprocess.TimeoutExpired:
        return C2PAResult(
            found=True,
            valid=False,
            tool=exe,
            raw_json=None,
            manifest_store_raw=None,
            manifest_store_error=None,
            claims=None,
            validation=None,
            error="timeout",
        )
    except Exception as e:
        return C2PAResult(
            found=True,
            valid=False,
            tool=exe,
            raw_json=None,
            manifest_store_raw=None,
            manifest_store_error=None,
            claims=None,
            validation=None,
            error=str(e),
        )
