from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.datasets.genre_catalog import MIDI_EXTENSIONS
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality


def clean_midi_dataset(
    config: EngineConfig,
    *,
    source_dir: Path,
    output_name: str = "clean_midis",
    min_duration_seconds: float = 1.0,
    max_duration_seconds: float = 240.0,
    min_notes: int = 4,
    min_quality_score: float = 0.05,
    deduplicate: bool = True,
) -> dict[str, Any]:
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Carpeta MIDI no encontrada: {root}")

    dataset_id = create_id(output_name, prefix="clean")
    output_dir = config.datasets_dir / "clean" / dataset_id
    midi_dir = output_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in MIDI_EXTENSIONS:
            continue
        try:
            digest = _file_hash(path)
            if deduplicate and digest in seen_hashes:
                rejected.append({"path": str(path), "reason": "duplicate_hash"})
                continue
            metrics = analyze_midi_quality(path)
            reason = _rejection_reason(
                metrics,
                min_duration_seconds=min_duration_seconds,
                max_duration_seconds=max_duration_seconds,
                min_notes=min_notes,
                min_quality_score=min_quality_score,
            )
            if reason:
                rejected.append({"path": str(path), "reason": reason, "metrics": metrics})
                continue
            seen_hashes.add(digest)
            target = midi_dir / f"{len(accepted) + 1:06d}_{path.stem}.mid"
            shutil.copy2(path, target)
            accepted.append(
                {
                    "source_midi": str(path),
                    "clean_midi": str(target),
                    "sha256": digest,
                    "metrics": metrics,
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"error: {exc}"})

    manifest_path = output_dir / "manifest.json"
    payload = {
        "schema_version": "clean-midi-dataset-v1",
        "dataset_id": dataset_id,
        "output_name": output_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(root),
        "output_dir": str(output_dir),
        "filters": {
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "min_notes": min_notes,
            "min_quality_score": min_quality_score,
            "deduplicate": deduplicate,
        },
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _rejection_reason(
    metrics: dict[str, Any],
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
    min_notes: int,
    min_quality_score: float,
) -> str | None:
    duration = float(metrics.get("duration_seconds", 0.0))
    if duration < min_duration_seconds:
        return "too_short"
    if duration > max_duration_seconds:
        return "too_long"
    if int(metrics.get("note_count", 0)) < min_notes:
        return "too_few_notes"
    if float(metrics.get("quality_score", 0.0)) < min_quality_score:
        return "quality_below_threshold"
    return None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
