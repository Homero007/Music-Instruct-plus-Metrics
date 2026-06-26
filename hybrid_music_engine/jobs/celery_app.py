from __future__ import annotations

from celery import Celery

from hybrid_music_engine.core.config import EngineConfig


config = EngineConfig.from_env()

celery_app = Celery(
    "hybrid_music_engine",
    broker=config.celery_broker_url,
    backend=config.celery_result_backend,
    include=["hybrid_music_engine.jobs.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
