from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EngineConfig:
    project_root: Path
    data_dir: Path
    projects_dir: Path
    jobs_dir: Path
    datasets_dir: Path
    tokens_dir: Path
    artifacts_dir: Path
    assets_dir: Path
    soundfonts_dir: Path
    default_soundfont_path: Path
    celery_broker_url: str
    celery_result_backend: str
    job_backend: str = "local"
    require_celery: bool = False
    vst_render_command: str | None = None
    fluidsynth_binary: str = "fluidsynth"
    ffmpeg_binary: str = "ffmpeg"
    default_sample_rate: int = 48000
    default_channels: int = 2

    @classmethod
    def for_project_root(cls, root: Path) -> "EngineConfig":
        data_dir = root / "data"
        broker = os.getenv("HYBRID_ENGINE_BROKER_URL", "redis://localhost:6379/0")
        backend = os.getenv("HYBRID_ENGINE_RESULT_BACKEND", broker)
        assets_dir = root / "assets"
        soundfont_path = Path(
            os.getenv("HYBRID_SOUNDFONT_PATH", assets_dir / "soundfonts" / "default.sf2")
        )
        return cls(
            project_root=root,
            data_dir=data_dir,
            projects_dir=data_dir / "projects",
            jobs_dir=data_dir / "jobs",
            datasets_dir=data_dir / "datasets",
            tokens_dir=data_dir / "tokens",
            artifacts_dir=data_dir / "artifacts",
            assets_dir=assets_dir,
            soundfonts_dir=assets_dir / "soundfonts",
            default_soundfont_path=soundfont_path,
            celery_broker_url=broker,
            celery_result_backend=backend,
            job_backend=os.getenv("HYBRID_ENGINE_JOB_BACKEND", "local"),
            require_celery=os.getenv("HYBRID_ENGINE_REQUIRE_CELERY", "0") == "1",
            vst_render_command=os.getenv("HYBRID_VST_RENDER_COMMAND"),
            fluidsynth_binary=os.getenv("HYBRID_FLUIDSYNTH_BIN", "fluidsynth"),
            ffmpeg_binary=os.getenv("HYBRID_FFMPEG_BIN", "ffmpeg"),
            default_sample_rate=int(os.getenv("HYBRID_ENGINE_SAMPLE_RATE", "48000")),
            default_channels=int(os.getenv("HYBRID_ENGINE_CHANNELS", "2")),
        )

    @classmethod
    def from_env(cls) -> "EngineConfig":
        root = Path(os.getenv("HYBRID_ENGINE_ROOT", Path(__file__).resolve().parents[2]))
        return cls.for_project_root(root)

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.soundfonts_dir.mkdir(parents=True, exist_ok=True)


def resolve_inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(f"Ruta fuera del proyecto: {path}")
    return resolved_path
