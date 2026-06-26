from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id


MIDI_EXTENSIONS = {".mid", ".midi"}


def midi_duration_seconds(path: Path) -> float:
    try:
        from mido import MidiFile, tick2second
    except ImportError as exc:
        raise RuntimeError("mido es necesario para leer duración de MIDI.") from exc

    midi = MidiFile(path)
    tempo = 500000
    max_seconds = 0.0
    for track in midi.tracks:
        absolute_ticks = 0
        for message in track:
            absolute_ticks += int(message.time)
            if message.type == "set_tempo":
                tempo = int(message.tempo)
            seconds = tick2second(absolute_ticks, midi.ticks_per_beat, tempo)
            max_seconds = max(max_seconds, float(seconds))
    return round(max_seconds, 4)


def build_genre_catalog(
    config: EngineConfig,
    *,
    source_dir: Path,
    genres: list[str],
    clips_per_genre: int = 200,
    max_duration_seconds: float = 10.0,
    catalog_name: str = "genre_catalog",
    source_label: str = "source_3",
) -> dict[str, Any]:
    normalized_genres = [genre.strip() for genre in genres if genre.strip()]
    if not normalized_genres:
        raise RuntimeError("Debes indicar al menos un género.")
    if len(set(normalized_genres)) != len(normalized_genres):
        raise RuntimeError("Los nombres de género no deben repetirse.")
    if clips_per_genre <= 0:
        raise RuntimeError("clips_per_genre debe ser mayor que cero.")

    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Carpeta fuente no encontrada: {root}")

    catalog_id = create_id(catalog_name, prefix="catalog")
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for genre in normalized_genres:
        genre_dir = root / genre
        if not genre_dir.exists():
            raise RuntimeError(f"No existe la carpeta del género '{genre}': {genre_dir}")
        selected = 0
        for midi_path in sorted(genre_dir.rglob("*")):
            if midi_path.suffix.lower() not in MIDI_EXTENSIONS:
                continue
            try:
                duration = midi_duration_seconds(midi_path)
            except (OSError, ValueError, RuntimeError) as exc:
                rejected.append(
                    {
                        "genre": genre,
                        "path": str(midi_path),
                        "reason": f"midi_read_error: {exc}",
                    }
                )
                continue
            if duration > max_duration_seconds:
                rejected.append(
                    {
                        "genre": genre,
                        "path": str(midi_path),
                        "duration_seconds": duration,
                        "reason": "duration_exceeds_limit",
                    }
                )
                continue
            clip_id = create_id(f"{genre}-{midi_path.stem}", prefix="clip")
            entries.append(
                {
                    "clip_id": clip_id,
                    "genre": genre,
                    "source_midi": str(midi_path),
                    "duration_seconds": duration,
                }
            )
            selected += 1
            if selected >= clips_per_genre:
                break
        counts[genre] = selected

    now = datetime.now().isoformat(timespec="seconds")
    catalog = {
        "schema_version": "genre-midi-catalog-v1",
        "catalog_id": catalog_id,
        "catalog_name": catalog_name,
        "source_label": source_label,
        "source_dir": str(root),
        "created_at": now,
        "genres": normalized_genres,
        "genre_count": len(normalized_genres),
        "clips_per_genre_requested": clips_per_genre,
        "max_duration_seconds": max_duration_seconds,
        "counts": counts,
        "total_clips": len(entries),
        "entries": entries,
        "rejected": rejected,
    }
    output_dir = config.datasets_dir / "catalogs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{catalog_id}.json"
    catalog["path"] = str(output_path)
    output_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def load_catalog(path: Path) -> dict[str, Any]:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.exists():
        raise RuntimeError(f"Catálogo no encontrado: {catalog_path}")
    return json.loads(catalog_path.read_text(encoding="utf-8"))
