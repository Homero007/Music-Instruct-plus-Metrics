from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id


PROJECT_SUBDIRS = [
    "source",
    "stems",
    "midis",
    "features",
    "embeddings",
    "generated",
    "renders",
    "reports",
]


@dataclass
class ProjectManifest:
    project_id: str
    name: str
    created_at: str
    updated_at: str
    status: str = "created"
    source: dict[str, Any] = field(default_factory=dict)
    stems: dict[str, Any] = field(default_factory=dict)
    midis: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, Any] = field(default_factory=dict)
    generated: dict[str, Any] = field(default_factory=dict)
    renders: dict[str, Any] = field(default_factory=dict)


def create_project(config: EngineConfig, name: str) -> ProjectManifest:
    config.ensure_directories()
    now = datetime.now().isoformat(timespec="seconds")
    project_id = create_id(name, prefix="project")
    project_dir = project_path(config, project_id)
    project_dir.mkdir(parents=True, exist_ok=False)
    for subdir in PROJECT_SUBDIRS:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest(project_id=project_id, name=name, created_at=now, updated_at=now)
    save_manifest(config, manifest)
    return manifest


def project_path(config: EngineConfig, project_id: str) -> Path:
    return config.projects_dir / project_id


def manifest_path(config: EngineConfig, project_id: str) -> Path:
    return project_path(config, project_id) / "manifest.json"


def load_manifest(config: EngineConfig, project_id: str) -> ProjectManifest:
    path = manifest_path(config, project_id)
    if not path.exists():
        raise RuntimeError("Proyecto no encontrado.")
    return ProjectManifest(**json.loads(path.read_text(encoding="utf-8")))


def save_manifest(config: EngineConfig, manifest: ProjectManifest) -> None:
    manifest.updated_at = datetime.now().isoformat(timespec="seconds")
    path = manifest_path(config, manifest.project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def list_projects(config: EngineConfig) -> list[dict[str, Any]]:
    config.ensure_directories()
    rows: list[dict[str, Any]] = []
    for path in sorted(config.projects_dir.glob("*/manifest.json"), reverse=True):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows
