from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pixspector.config import Config, find_defaults_path
from pixspector.pipeline import analyze_single_image

from .job_store import JobStore

APP_TITLE = "Pixspector API"
DEFAULT_JOB_DIR = Path(os.getenv("PIXSPECTOR_JOB_DIR", "var/pixspector_jobs")).resolve()
MAX_WORKERS = int(os.getenv("PIXSPECTOR_WORKERS", "2"))

app = FastAPI(title=APP_TITLE, version="0.1.0")
store = JobStore(DEFAULT_JOB_DIR)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    context: Optional[str] = Form(default=None),
) -> dict:
    job = store.create_job(file.filename or "upload", context)
    job_id = job["job_id"]
    input_path = Path(job["input"]["path"])
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    executor.submit(_run_pipeline, job_id, input_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job.get("error"),
    }


@app.get("/result/{job_id}")
async def result(job_id: str) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
    report_path = Path(job["output"]["report_path"])
    if not report_path.exists():
        raise HTTPException(status_code=500, detail="Report missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "job_id": job_id,
        "report": report,
        "evidence_links": job["output"]["evidence_links"],
    }


def _run_pipeline(job_id: str, input_path: Path) -> None:
    store.set_status(job_id, "running")
    job_paths = store.job_paths(job_id)
    try:
        cfg = Config.load(find_defaults_path(Path(__file__).resolve()))
        result = analyze_single_image(
            input_path,
            cfg,
            out_dir=job_paths.output_dir,
            want_pdf=False,
        )
        report_path = job_paths.output_dir / f"{input_path.stem}_report.json"
        artifacts_dir = job_paths.output_dir / f"{input_path.stem}_artifacts"
        evidence_links = _collect_evidence_links(artifacts_dir)
        store.set_result(job_id, report_path, artifacts_dir, evidence_links)
        _ = result
    except Exception as exc:  # pylint: disable=broad-except
        store.set_status(job_id, "failed", error=str(exc))


def _collect_evidence_links(artifacts_dir: Path) -> list[str]:
    if not artifacts_dir.exists():
        return []
    files = [path for path in artifacts_dir.rglob("*") if path.is_file()]
    return [str(path) for path in sorted(files)]
