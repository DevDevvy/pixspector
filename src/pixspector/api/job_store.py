from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class JobPaths:
    job_dir: Path
    input_dir: Path
    output_dir: Path
    job_file: Path


class JobStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_job(self, filename: str, context: Optional[str]) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        paths = self._init_job_dirs(job_id)
        safe_name = Path(filename or "upload").name
        created_at = _utc_now()
        record = {
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "context": context,
            "input": {
                "filename": safe_name,
                "path": str(paths.input_dir / safe_name),
            },
            "output": {
                "report_path": None,
                "artifacts_dir": None,
                "evidence_links": [],
            },
            "error": None,
        }
        self._write(paths.job_file, record)
        return record

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job_file = self._job_file(job_id)
        if not job_file.exists():
            return None
        return self._read(job_file)

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            job_file = self._job_file(job_id)
            record = self._read(job_file)
            record.update(updates)
            record["updated_at"] = _utc_now()
            self._write(job_file, record)
            return record

    def set_status(self, job_id: str, status: str, error: Optional[str] = None) -> Dict[str, Any]:
        return self.update_job(job_id, {"status": status, "error": error})

    def set_result(
        self,
        job_id: str,
        report_path: Path,
        artifacts_dir: Path,
        evidence_links: list[str],
    ) -> Dict[str, Any]:
        return self.update_job(
            job_id,
            {
                "status": "completed",
                "output": {
                    "report_path": str(report_path),
                    "artifacts_dir": str(artifacts_dir),
                    "evidence_links": evidence_links,
                },
                "error": None,
            },
        )

    def job_paths(self, job_id: str) -> JobPaths:
        job_dir = self.base_dir / job_id
        return JobPaths(
            job_dir=job_dir,
            input_dir=job_dir / "input",
            output_dir=job_dir / "output",
            job_file=job_dir / "job.json",
        )

    def _init_job_dirs(self, job_id: str) -> JobPaths:
        paths = self.job_paths(job_id)
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        return paths

    def _job_file(self, job_id: str) -> Path:
        return self.base_dir / job_id / "job.json"

    def _read(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
