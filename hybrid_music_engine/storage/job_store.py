from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    project_id: str | None = None
    progress: float | None = None
    stage: str = "created"
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


def create_job(config: EngineConfig, kind: str, project_id: str | None = None) -> JobRecord:
    config.ensure_directories()
    now = datetime.now().isoformat(timespec="seconds")
    job = JobRecord(
        job_id=create_id(kind, prefix="job"),
        kind=kind,
        project_id=project_id,
        status="queued",
        created_at=now,
        updated_at=now,
        events=[{"time": now, "stage": "queued", "message": "Job creado."}],
    )
    save_job(config, job)
    return job


def job_path(config: EngineConfig, job_id: str) -> Path:
    return config.jobs_dir / f"{job_id}.json"


def load_job(config: EngineConfig, job_id: str) -> JobRecord:
    path = job_path(config, job_id)
    if not path.exists():
        raise RuntimeError("Job no encontrado.")
    return JobRecord(**json.loads(path.read_text(encoding="utf-8")))


def save_job(config: EngineConfig, job: JobRecord) -> None:
    job.updated_at = datetime.now().isoformat(timespec="seconds")
    path = job_path(config, job.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")


def update_job(
    config: EngineConfig,
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    progress: float | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> JobRecord:
    job = load_job(config, job_id)
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    job.events.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "stage": job.stage,
            "message": job.message,
            "progress": job.progress,
            "status": job.status,
        }
    )
    save_job(config, job)
    return job


def list_jobs(config: EngineConfig) -> list[dict[str, Any]]:
    config.ensure_directories()
    rows: list[dict[str, Any]] = []
    for path in sorted(config.jobs_dir.glob("*.json"), reverse=True):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows
