from __future__ import annotations

import json
import random
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.datasets.genre_catalog import MIDI_EXTENSIONS, load_catalog, midi_duration_seconds


def augment_midi_dataset(
    config: EngineConfig,
    *,
    catalog_path: Path | None = None,
    source_dir: Path | None = None,
    output_name: str = "augmented_midis",
    transpose_steps: list[int] | None = None,
    velocity_jitter: int = 8,
    timing_jitter_ticks: int = 12,
    quantize_step_ticks: int | None = None,
    tempo_scale: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    if not catalog_path and not source_dir:
        raise RuntimeError("Indica catalog_path o source_dir para augmentación MIDI.")
    if tempo_scale <= 0:
        raise RuntimeError("tempo_scale debe ser mayor que cero.")

    rng = random.Random(seed)
    transpose_values = transpose_steps if transpose_steps is not None else [-2, 0, 2]
    if not transpose_values:
        raise RuntimeError("transpose_steps debe contener al menos un valor.")

    source_entries = _entries_from_catalog(catalog_path) if catalog_path else _entries_from_dir(source_dir)
    if not source_entries:
        raise RuntimeError("No se encontraron MIDIs para augmentar.")

    run_id = create_id(output_name, prefix="augment")
    output_root = config.datasets_dir / "augmented" / run_id
    midi_root = output_root / "midis"
    midi_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in source_entries:
        source_midi = Path(str(entry["source_midi"])).expanduser()
        genre = str(entry.get("genre") or "unknown")
        for semitones in transpose_values:
            variant_id = create_id(f"{genre}-{source_midi.stem}-t{semitones}", prefix="clip")
            target_dir = midi_root / genre
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{variant_id}.mid"
            try:
                _augment_one_midi(
                    source_midi,
                    target_path,
                    semitones=semitones,
                    velocity_jitter=velocity_jitter,
                    timing_jitter_ticks=timing_jitter_ticks,
                    quantize_step_ticks=quantize_step_ticks,
                    tempo_scale=tempo_scale,
                    rng=rng,
                )
                entries.append(
                    {
                        "clip_id": variant_id,
                        "genre": genre,
                        "source_midi": str(target_path),
                        "original_midi": str(source_midi),
                        "duration_seconds": midi_duration_seconds(target_path),
                        "augmentation": {
                            "transpose_semitones": semitones,
                            "velocity_jitter": velocity_jitter,
                            "timing_jitter_ticks": timing_jitter_ticks,
                            "quantize_step_ticks": quantize_step_ticks,
                            "tempo_scale": tempo_scale,
                        },
                    }
                )
            except (OSError, ValueError, RuntimeError) as exc:
                rejected.append(
                    {
                        "source_midi": str(source_midi),
                        "genre": genre,
                        "transpose_semitones": semitones,
                        "reason": str(exc),
                    }
                )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["genre"]] = counts.get(entry["genre"], 0) + 1

    now = datetime.now().isoformat(timespec="seconds")
    catalog = {
        "schema_version": "augmented-midi-catalog-v1",
        "catalog_id": run_id,
        "catalog_name": output_name,
        "created_at": now,
        "source_catalog_path": str(catalog_path) if catalog_path else None,
        "source_dir": str(source_dir) if source_dir else None,
        "genres": sorted(counts),
        "counts": counts,
        "total_clips": len(entries),
        "entries": entries,
        "rejected": rejected,
    }
    catalog_path_out = output_root / "catalog.json"
    summary_path = output_root / "augmentation_summary.json"
    catalog["path"] = str(catalog_path_out)
    catalog_path_out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id,
        "created_at": now,
        "output_root": str(output_root),
        "catalog_path": str(catalog_path_out),
        "total_sources": len(source_entries),
        "total_augmented": len(entries),
        "total_rejected": len(rejected),
        "counts": counts,
        "settings": {
            "transpose_steps": transpose_values,
            "velocity_jitter": velocity_jitter,
            "timing_jitter_ticks": timing_jitter_ticks,
            "quantize_step_ticks": quantize_step_ticks,
            "tempo_scale": tempo_scale,
            "seed": seed,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**summary, "catalog": catalog, "summary_path": str(summary_path)}


def _entries_from_catalog(catalog_path: Path | None) -> list[dict[str, Any]]:
    if not catalog_path:
        return []
    catalog = load_catalog(catalog_path)
    entries = []
    for entry in catalog.get("entries", []):
        source = entry.get("source_midi") or entry.get("path")
        if source:
            entries.append({**entry, "source_midi": source})
    return entries


def _entries_from_dir(source_dir: Path | None) -> list[dict[str, Any]]:
    if not source_dir:
        return []
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Carpeta MIDI no encontrada: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in MIDI_EXTENSIONS:
            continue
        genre = path.parent.name if path.parent != root else "unknown"
        entries.append({"genre": genre, "source_midi": str(path)})
    return entries


def _augment_one_midi(
    source_midi: Path,
    target_path: Path,
    *,
    semitones: int,
    velocity_jitter: int,
    timing_jitter_ticks: int,
    quantize_step_ticks: int | None,
    tempo_scale: float,
    rng: random.Random,
) -> None:
    try:
        from mido import MidiFile
    except ImportError as exc:
        raise RuntimeError("mido es necesario para augmentación MIDI.") from exc

    midi = MidiFile(source_midi)
    for track in midi.tracks:
        absolute = 0
        events = []
        for message in track:
            absolute += int(message.time)
            msg = copy(message)
            if msg.type == "set_tempo":
                msg.tempo = max(int(msg.tempo / tempo_scale), 1)
            if msg.type in {"note_on", "note_off"}:
                if int(getattr(msg, "channel", 0)) != 9:
                    msg.note = max(0, min(127, int(msg.note) + semitones))
                if msg.type == "note_on" and msg.velocity > 0 and velocity_jitter > 0:
                    msg.velocity = max(
                        1,
                        min(127, int(msg.velocity) + rng.randint(-velocity_jitter, velocity_jitter)),
                    )
            event_tick = absolute
            if msg.type in {"note_on", "note_off"}:
                if quantize_step_ticks and quantize_step_ticks > 0:
                    event_tick = int(round(event_tick / quantize_step_ticks) * quantize_step_ticks)
                if timing_jitter_ticks > 0:
                    event_tick += rng.randint(-timing_jitter_ticks, timing_jitter_ticks)
                event_tick = max(event_tick, 0)
            events.append((event_tick, msg))

        events.sort(key=lambda item: item[0])
        last_tick = 0
        track.clear()
        for tick, msg in events:
            msg.time = max(int(tick) - last_tick, 0)
            track.append(msg)
            last_tick = max(int(tick), last_tick)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(target_path)
