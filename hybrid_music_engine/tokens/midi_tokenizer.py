from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.datasets.genre_catalog import MIDI_EXTENSIONS, load_catalog, midi_duration_seconds
from hybrid_music_engine.generation.multitrack_writer import classify_midi_event_layer


def tokenize_midi_file(
    midi_path: Path,
    *,
    genre: str | None = None,
    clip_id: str | None = None,
    include_structure: bool = True,
    tokenizer_mode: str = "remi",
) -> dict[str, Any]:
    try:
        from mido import MidiFile
    except ImportError as exc:
        raise RuntimeError("mido es necesario para tokenizar MIDI.") from exc

    path = Path(midi_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"MIDI no encontrado: {path}")

    midi = MidiFile(path)
    if tokenizer_mode == "remi":
        tokens = _tokenize_midi_remi(midi)
        tokenizer_name = "remi-lite-midi-tokenizer"
    elif tokenizer_mode in {"structured", "event"}:
        tokens = _tokenize_midi_events(midi, include_structure=include_structure)
        tokenizer_name = (
            "structured-midi-event-tokenizer"
            if include_structure and tokenizer_mode == "structured"
            else "deterministic-midi-event-tokenizer"
        )
    else:
        raise RuntimeError("tokenizer_mode debe ser remi, structured o event.")

    return {
        "schema_version": "midi-token-json-v1",
        "source_midi": str(path),
        "clip_id": clip_id or create_id(path.stem, prefix="clip"),
        "genre": genre,
        "duration_seconds": midi_duration_seconds(path),
        "tokenizer": {
            "name": tokenizer_name,
            "mode": tokenizer_mode,
            "time_unit": "midi_delta_ticks" if tokenizer_mode != "remi" else "bar_position_duration",
            "velocity_buckets": 16,
            "structure_tokens": include_structure or tokenizer_mode == "remi",
            "position_slots_per_beat": 4,
        },
        "token_count": len(tokens),
        "tokens": tokens,
    }


def _tokenize_midi_events(midi, *, include_structure: bool) -> list[str]:
    tokens: list[str] = [f"ticks_per_beat:{midi.ticks_per_beat}"]
    for track_index, track in enumerate(midi.tracks):
        tokens.append(f"track_start:{track_index}")
        absolute_ticks = 0
        previous_position: tuple[int, int, int] | None = None
        for message in track:
            absolute_ticks += int(message.time)
            if include_structure:
                position = _musical_position(absolute_ticks, midi.ticks_per_beat)
                if position != previous_position:
                    bar, beat, slot = position
                    tokens.append(f"bar:{bar}")
                    tokens.append(f"beat:{beat}")
                    tokens.append(f"pos:{slot}")
                    previous_position = position
            if message.time:
                tokens.append(f"dt:{int(message.time)}")
            if message.type == "set_tempo":
                tokens.append(f"tempo:{int(message.tempo)}")
            elif message.type == "program_change":
                tokens.append(
                    f"program:{int(message.channel)}:{int(message.program)}"
                )
            elif message.type in {"note_on", "note_off"}:
                velocity = int(getattr(message, "velocity", 0))
                token_type = "note_on" if message.type == "note_on" and velocity > 0 else "note_off"
                velocity_bucket = min(15, max(0, velocity // 8))
                tokens.append(
                    f"{token_type}:{int(message.channel)}:{int(message.note)}:{velocity_bucket}"
                )
            elif message.type == "control_change":
                tokens.append(
                    f"control:{int(message.channel)}:{int(message.control)}:{int(message.value)}"
                )
            elif message.type == "pitchwheel":
                tokens.append(f"pitchwheel:{int(message.channel)}:{int(message.pitch)}")
        tokens.append(f"track_end:{track_index}")

    return tokens


def _tokenize_midi_remi(midi) -> list[str]:
    tokens: list[str] = [f"ticks_per_beat:{midi.ticks_per_beat}"]
    events: list[dict[str, Any]] = []
    tempo = 500000
    for track_index, track in enumerate(midi.tracks):
        absolute_ticks = 0
        current_program = {channel: 0 for channel in range(16)}
        active: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for message in track:
            absolute_ticks += int(message.time)
            if message.type == "set_tempo":
                tempo = int(message.tempo)
            elif message.type == "program_change":
                current_program[int(message.channel)] = int(message.program)
            elif message.type == "note_on" and message.velocity > 0:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                active.setdefault(key, []).append(
                    (absolute_ticks, int(message.velocity), current_program.get(key[0], 0))
                )
            elif message.type in {"note_off", "note_on"}:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                starts = active.get(key)
                if not starts:
                    continue
                start_tick, velocity, program = starts.pop(0)
                duration = max(absolute_ticks - start_tick, 1)
                layer = classify_midi_event_layer(channel=key[0], note=key[1], program=program)
                events.append(
                    {
                        "tick": start_tick,
                        "track": track_index,
                        "channel": key[0],
                        "note": key[1],
                        "velocity": velocity,
                        "program": program,
                        "duration": duration,
                        "layer": layer,
                    }
                )
    tokens.append(f"tempo:{tempo}")
    last_bar: int | None = None
    last_position: int | None = None
    last_program: tuple[int, int] | None = None
    for event in sorted(events, key=lambda item: (item["tick"], item["track"], item["note"])):
        bar, beat, slot = _musical_position(int(event["tick"]), midi.ticks_per_beat)
        position = (beat * 4) + slot
        if bar != last_bar:
            tokens.append(f"bar:{bar}")
            last_bar = bar
            last_position = None
        if position != last_position:
            tokens.append(f"position:{position}")
            last_position = position
        program_key = (int(event["channel"]), int(event["program"]))
        if program_key != last_program:
            tokens.append(f"program:{program_key[0]}:{program_key[1]}")
            last_program = program_key
        velocity_bucket = min(15, max(0, int(event["velocity"]) // 8))
        tokens.append(f"layer:{event['layer']}")
        tokens.append(
            "note:"
            f"{int(event['channel'])}:"
            f"{int(event['note'])}:"
            f"{velocity_bucket}:"
            f"{int(event['duration'])}"
        )
    return tokens


def tokenize_catalog_to_zip(
    config: EngineConfig,
    *,
    catalog_path: Path,
    token_set_name: str = "input_tokens",
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    token_set_id = create_id(token_set_name, prefix="tokens")
    output_dir = config.tokens_dir / "input" / token_set_id
    token_files_dir = output_dir / "genres"
    token_files_dir.mkdir(parents=True, exist_ok=True)

    token_entries: list[dict[str, Any]] = []
    for entry in catalog.get("entries", []):
        genre = str(entry["genre"])
        genre_dir = token_files_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        token_payload = tokenize_midi_file(
            Path(str(entry["source_midi"])),
            genre=genre,
            clip_id=str(entry["clip_id"]),
        )
        token_path = genre_dir / f"{entry['clip_id']}.tokens.json"
        token_payload["path"] = str(token_path)
        token_path.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
        token_entries.append(
            {
                "clip_id": token_payload["clip_id"],
                "genre": genre,
                "token_count": token_payload["token_count"],
                "duration_seconds": token_payload["duration_seconds"],
                "path": str(token_path),
            }
        )

    manifest = {
        "schema_version": "token-set-manifest-v1",
        "kind": "input",
        "token_set_id": token_set_id,
        "token_set_name": token_set_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "catalog_path": str(Path(catalog_path).expanduser().resolve()),
        "catalog_id": catalog.get("catalog_id"),
        "genres": catalog.get("genres", []),
        "counts": _counts_by_genre(token_entries),
        "total_files": len(token_entries),
        "entries": token_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest["path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = _zip_directory(output_dir, config.artifacts_dir / f"{token_set_id}.zip")
    manifest["zip_path"] = str(zip_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_output_tokens_to_zip(
    config: EngineConfig,
    *,
    source_dir: Path,
    export_name: str = "mixed_output_tokens",
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Ruta de salida no encontrada: {root}")
    source_paths = [root] if root.is_file() else sorted(root.rglob("*"))
    token_sidecars = {
        path.with_suffix("").with_suffix(".mid").stem
        for path in source_paths
        if path.suffix.lower() == ".json" and path.name.endswith(".tokens.json")
    }

    token_set_id = create_id(export_name, prefix="tokens")
    output_dir = config.tokens_dir / "output" / token_set_id
    genres_dir = output_dir / "genres"
    genres_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for index, path in enumerate(source_paths, start=1):
        if path.suffix.lower() == ".json" and path.name.endswith(".tokens.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            genre = _genre_for_token_payload(payload, path)
            target = _token_target_path(genres_dir, genre, payload, path, index)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            entries.append(
                {
                    "source": str(path),
                    "path": str(target),
                    "genre": genre,
                    "token_count": payload.get("token_count"),
                    "duration_seconds": payload.get("duration_seconds"),
                }
            )
        elif path.suffix.lower() in MIDI_EXTENSIONS:
            if path.stem in token_sidecars:
                continue
            payload = tokenize_midi_file(path)
            genre = _genre_for_token_payload(payload, path)
            target = _token_target_path(genres_dir, genre, payload, path, index)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload["path"] = str(target)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            entries.append(
                {
                    "source": str(path),
                    "path": str(target),
                    "genre": genre,
                    "token_count": payload["token_count"],
                    "duration_seconds": payload["duration_seconds"],
                }
            )

    if not entries:
        raise RuntimeError("No se encontraron archivos .tokens.json ni MIDI para exportar.")

    manifest = {
        "schema_version": "token-set-manifest-v1",
        "kind": "output",
        "token_set_id": token_set_id,
        "export_name": export_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(root),
        "requested_duration_seconds": duration_seconds,
        "genres": sorted(_counts_by_genre(entries)),
        "counts": _counts_by_genre(entries),
        "total_files": len(entries),
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest["path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = _zip_directory(output_dir, config.artifacts_dir / f"{token_set_id}.zip")
    manifest["zip_path"] = str(zip_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_token_manifest_to_zip(
    config: EngineConfig,
    *,
    token_manifest_path: Path,
    export_name: str = "input_tokens",
) -> dict[str, Any]:
    manifest_source = Path(token_manifest_path).expanduser().resolve()
    if not manifest_source.exists():
        raise RuntimeError(f"Manifest de tokens no encontrado: {manifest_source}")
    source_payload = json.loads(manifest_source.read_text(encoding="utf-8"))
    entries = source_payload.get("entries", [])
    if not entries:
        raise RuntimeError("El manifest de tokens no contiene entradas.")

    token_set_id = create_id(export_name, prefix="tokens")
    output_dir = config.tokens_dir / "input" / token_set_id
    genres_dir = output_dir / "genres"
    genres_dir.mkdir(parents=True, exist_ok=True)

    exported_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        genre = str(entry.get("genre") or _genre_for_token_payload(payload, path))
        target = _token_target_path(genres_dir, genre, payload, path, index)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        exported_entries.append(
            {
                "source": str(path),
                "path": str(target),
                "clip_id": payload.get("clip_id") or entry.get("clip_id"),
                "genre": genre,
                "token_count": payload.get("token_count") or entry.get("token_count"),
                "duration_seconds": payload.get("duration_seconds") or entry.get("duration_seconds"),
            }
        )

    if not exported_entries:
        raise RuntimeError("No se encontraron archivos de tokens válidos dentro del manifest.")

    manifest = {
        "schema_version": "token-set-manifest-v1",
        "kind": "input",
        "token_set_id": token_set_id,
        "token_set_name": export_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest_path": str(manifest_source),
        "genres": sorted(_counts_by_genre(exported_entries)),
        "counts": _counts_by_genre(exported_entries),
        "total_files": len(exported_entries),
        "entries": exported_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest["path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = _zip_directory(output_dir, config.artifacts_dir / f"{token_set_id}.zip")
    manifest["zip_path"] = str(zip_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def create_generation_plan(
    config: EngineConfig,
    *,
    project_id: str | None,
    duration_seconds: float,
    output_name: str = "generated_track",
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise RuntimeError("duration_seconds debe ser mayor que cero.")
    plan_id = create_id(output_name, prefix="generation")
    output_dir = config.data_dir / "generation_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{plan_id}.json"
    payload = {
        "schema_version": "generation-plan-v1",
        "plan_id": plan_id,
        "project_id": project_id,
        "output_name": output_name,
        "duration_seconds": duration_seconds,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "path": str(path),
        "status": "planned",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _counts_by_genre(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        genre = str(entry["genre"])
        counts[genre] = counts.get(genre, 0) + 1
    return counts


def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return zip_path


def _genre_for_token_payload(payload: dict[str, Any], source_path: Path) -> str:
    value = payload.get("genre") or payload.get("condition_genre")
    if not value and payload.get("embedding_path"):
        value = "hybrid"
    if not value:
        parts = [part.lower() for part in source_path.parts]
        if "genres" in parts:
            index = parts.index("genres")
            if index + 1 < len(source_path.parts):
                value = source_path.parts[index + 1]
    return _safe_folder_name(str(value or "generated"))


def _token_target_path(
    genres_dir: Path,
    genre: str,
    payload: dict[str, Any],
    source_path: Path,
    index: int,
) -> Path:
    raw_name = (
        payload.get("clip_id")
        or payload.get("generation_id")
        or payload.get("candidate_id")
        or source_path.stem
        or f"tokens_{index:04d}"
    )
    name = _safe_folder_name(str(raw_name))
    return genres_dir / _safe_folder_name(genre) / f"{index:04d}_{name}.tokens.json"


def _safe_folder_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip().lower())
    return cleaned.strip("_") or "unknown"


def _musical_position(absolute_ticks: int, ticks_per_beat: int) -> tuple[int, int, int]:
    beats_per_bar = 4
    slot_ticks = max(ticks_per_beat // 4, 1)
    beat_index = absolute_ticks // max(ticks_per_beat, 1)
    bar = int(beat_index // beats_per_bar)
    beat = int(beat_index % beats_per_bar)
    slot = int((absolute_ticks % max(ticks_per_beat, 1)) // slot_ticks)
    return bar, beat, slot
