from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.generation.multitrack_writer import (
    TRACK_NAMES,
    classify_midi_event_layer,
)
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality


def train_token_markov_model(
    config: EngineConfig,
    *,
    token_manifest_path: Path,
    model_name: str = "token_markov",
    order: int = 2,
) -> dict[str, Any]:
    if order < 1:
        raise RuntimeError("order debe ser mayor o igual a 1.")
    manifest = json.loads(Path(token_manifest_path).expanduser().read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if not entries:
        raise RuntimeError("El manifest de tokens no contiene entradas.")

    transitions: dict[tuple[str, ...], list[str]] = defaultdict(list)
    starts: list[list[str]] = []
    token_files = 0
    for entry in entries:
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = [str(token) for token in payload.get("tokens", [])]
        if len(tokens) <= order:
            continue
        starts.append(tokens[:order])
        token_files += 1
        for index in range(len(tokens) - order):
            key = tuple(tokens[index : index + order])
            transitions[key].append(tokens[index + order])

    if not transitions:
        raise RuntimeError("No hay suficientes tokens para entrenar el modelo.")

    model_id = create_id(model_name, prefix="tokenmodel")
    output_dir = config.data_dir / "models" / "tokens" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    serializable_transitions = {"\u241f".join(key): values for key, values in transitions.items()}
    model = {
        "schema_version": "token-markov-model-v1",
        "model_type": "markov",
        "model_id": model_id,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "order": order,
        "token_manifest_path": str(Path(token_manifest_path).expanduser().resolve()),
        "token_files": token_files,
        "states": len(serializable_transitions),
        "starts": starts,
        "transitions": serializable_transitions,
        "path": str(model_path),
    }
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def generate_tokens_from_model(
    config: EngineConfig,
    *,
    model_path: Path,
    duration_seconds: float,
    output_name: str = "generated",
    seed: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    condition_genre: str | None = None,
    feature_tokens: list[str] | None = None,
    embedding_path: Path | None = None,
    export_layers: bool = False,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise RuntimeError("duration_seconds debe ser mayor que cero.")
    model = json.loads(Path(model_path).expanduser().read_text(encoding="utf-8"))
    if model.get("schema_version") == "token-transformer-model-v1":
        from hybrid_music_engine.tokens.transformer_model import generate_tokens_from_transformer_model

        return generate_tokens_from_transformer_model(
            config,
            model_path=model_path,
            duration_seconds=duration_seconds,
            output_name=output_name,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            condition_genre=condition_genre,
            feature_tokens=feature_tokens,
            embedding_path=embedding_path,
            export_layers=export_layers,
        )
    order = int(model["order"])
    starts = model.get("starts", [])
    transitions = {
        tuple(key.split("\u241f")): [str(value) for value in values]
        for key, values in model.get("transitions", {}).items()
    }
    if not starts or not transitions:
        raise RuntimeError("Modelo de tokens inválido o vacío.")

    rng = random.Random(seed)
    target_tokens = max_tokens or max(int(duration_seconds * 32), order + 16)
    current = list(rng.choice(starts))
    generated = current[:]
    while len(generated) < target_tokens:
        key = tuple(generated[-order:])
        options = transitions.get(key)
        if not options:
            current = list(rng.choice(starts))
            generated.extend(current)
            continue
        generated.append(rng.choice(options))

    output_id = create_id(output_name, prefix="generated")
    output_dir = config.data_dir / "generated" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "tokens.json"
    midi_path = output_dir / f"{output_name}.mid"
    token_payload = {
        "schema_version": "generated-token-json-v1",
        "generation_id": output_id,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds_requested": duration_seconds,
        "seed": seed,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "condition_genre": condition_genre,
        "feature_tokens": feature_tokens or [],
        "embedding_path": str(Path(embedding_path).expanduser().resolve()) if embedding_path else None,
        "token_count": len(generated),
        "tokens": generated,
        "path": str(token_path),
        "midi_path": str(midi_path),
    }
    token_path.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
    tokens_to_midi(generated, midi_path, duration_seconds=duration_seconds)
    token_payload["metrics"] = analyze_midi_quality(midi_path)
    if export_layers:
        token_payload["layer_midis"] = tokens_to_layered_midis(
            generated,
            output_dir / "layers",
            duration_seconds=duration_seconds,
        )
        token_payload["layer_metrics"] = {
            name: analyze_midi_quality(Path(path))
            for name, path in token_payload["layer_midis"].items()
        }
        token_path.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
    else:
        token_path.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
    return token_payload


def tokens_to_midi(tokens: list[str], output_midi: Path, *, duration_seconds: float | None = None) -> Path:
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
    except ImportError as exc:
        raise RuntimeError("mido es necesario para convertir tokens a MIDI.") from exc

    ticks_per_beat = _first_ticks_per_beat(tokens) or 480
    tempo = _first_tempo(tokens) or bpm2tempo(120)
    if any(token.startswith("note:") for token in tokens):
        return _remi_tokens_to_midi(
            tokens,
            output_midi,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
            duration_seconds=duration_seconds,
        )
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

    pending_delta = 0
    absolute_ticks = 0
    wrote_notes = False
    for token in tokens:
        parts = token.split(":")
        kind = parts[0]
        if kind == "dt" and len(parts) >= 2:
            pending_delta += max(_safe_int(parts[1], 0), 0)
        elif kind == "program" and len(parts) >= 3:
            absolute_ticks += pending_delta
            track.append(
                Message(
                    "program_change",
                    channel=_clamp(_safe_int(parts[1], 0), 0, 15),
                    program=_clamp(_safe_int(parts[2], 0), 0, 127),
                    time=pending_delta,
                )
            )
            pending_delta = 0
        elif kind in {"note_on", "note_off"} and len(parts) >= 4:
            absolute_ticks += pending_delta
            velocity_bucket = _safe_int(parts[3], 0)
            velocity = 0 if kind == "note_off" else _clamp((velocity_bucket * 8) + 4, 1, 127)
            track.append(
                Message(
                    "note_on" if kind == "note_on" else "note_off",
                    channel=_clamp(_safe_int(parts[1], 0), 0, 15),
                    note=_clamp(_safe_int(parts[2], 60), 0, 127),
                    velocity=velocity,
                    time=pending_delta,
                )
            )
            pending_delta = 0
            wrote_notes = True

    if not wrote_notes:
        note_length = int(second2tick(0.25, ticks_per_beat, tempo))
        total_notes = max(int((duration_seconds or 4) * 2), 1)
        for index in range(total_notes):
            note = 60 + (index % 12)
            track.append(Message("note_on", channel=0, note=note, velocity=72, time=0 if index == 0 else note_length))
            track.append(Message("note_off", channel=0, note=note, velocity=0, time=note_length))
            absolute_ticks += note_length * 2

    _pad_track_to_duration(track, absolute_ticks, duration_seconds, ticks_per_beat=ticks_per_beat, tempo=tempo)

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_midi)
    return output_midi


def tokens_to_layered_midis(
    tokens: list[str],
    output_dir: Path,
    *,
    duration_seconds: float | None = None,
) -> dict[str, str]:
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
    except ImportError as exc:
        raise RuntimeError("mido es necesario para exportar MIDI por capas.") from exc

    ticks_per_beat = _first_ticks_per_beat(tokens) or 480
    tempo = _first_tempo(tokens) or bpm2tempo(120)
    if any(token.startswith("note:") for token in tokens):
        return _remi_tokens_to_layered_midis(
            tokens,
            output_dir,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
            duration_seconds=duration_seconds,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    midis: dict[str, MidiFile] = {}
    tracks: dict[str, MidiTrack] = {}
    last_ticks = {name: 0 for name in TRACK_NAMES}
    current_program = {channel: 0 for channel in range(16)}
    active_layers: dict[tuple[int, int], str] = {}

    for name in TRACK_NAMES:
        midi = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        midi.tracks.append(track)
        track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
        midis[name] = midi
        tracks[name] = track

    absolute_ticks = 0
    wrote_notes = {name: False for name in TRACK_NAMES}
    for token in tokens:
        parts = token.split(":")
        kind = parts[0]
        if kind == "dt" and len(parts) >= 2:
            absolute_ticks += max(_safe_int(parts[1], 0), 0)
            continue
        if kind == "program" and len(parts) >= 3:
            channel = _clamp(_safe_int(parts[1], 0), 0, 15)
            program = _clamp(_safe_int(parts[2], 0), 0, 127)
            current_program[channel] = program
            layer = classify_midi_event_layer(channel=channel, program=program)
            delta = max(absolute_ticks - last_ticks[layer], 0)
            tracks[layer].append(Message("program_change", channel=channel, program=program, time=delta))
            last_ticks[layer] = absolute_ticks
            continue
        if kind in {"note_on", "note_off"} and len(parts) >= 4:
            channel = _clamp(_safe_int(parts[1], 0), 0, 15)
            note = _clamp(_safe_int(parts[2], 60), 0, 127)
            velocity_bucket = _safe_int(parts[3], 0)
            velocity = 0 if kind == "note_off" else _clamp((velocity_bucket * 8) + 4, 1, 127)
            note_key = (channel, note)
            if kind == "note_on":
                layer = classify_midi_event_layer(
                    channel=channel,
                    note=note,
                    program=current_program.get(channel),
                )
                active_layers[note_key] = layer
                wrote_notes[layer] = True
            else:
                layer = active_layers.pop(
                    note_key,
                    classify_midi_event_layer(
                        channel=channel,
                        note=note,
                        program=current_program.get(channel),
                    ),
                )
            delta = max(absolute_ticks - last_ticks[layer], 0)
            tracks[layer].append(
                Message(
                    "note_on" if kind == "note_on" else "note_off",
                    channel=channel,
                    note=note,
                    velocity=velocity,
                    time=delta,
                )
            )
            last_ticks[layer] = absolute_ticks

    if not any(wrote_notes.values()):
        note_length = int(second2tick(0.25, ticks_per_beat, tempo))
        track = tracks["melody"]
        total_notes = max(int((duration_seconds or 4) * 2), 1)
        for index in range(total_notes):
            note = 60 + (index % 12)
            track.append(Message("note_on", channel=0, note=note, velocity=72, time=0 if index == 0 else note_length))
            track.append(Message("note_off", channel=0, note=note, velocity=0, time=note_length))
            last_ticks["melody"] += note_length * 2

    paths: dict[str, str] = {}
    for name, midi in midis.items():
        _pad_track_to_duration(
            tracks[name],
            last_ticks[name],
            duration_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        path = output_dir / f"{name}.mid"
        midi.save(path)
        paths[name] = str(path)
    return paths


def _remi_tokens_to_midi(
    tokens: list[str],
    output_midi: Path,
    *,
    ticks_per_beat: int,
    tempo: int,
    duration_seconds: float | None = None,
) -> Path:
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack, second2tick
    except ImportError as exc:
        raise RuntimeError("mido es necesario para convertir tokens REMI a MIDI.") from exc

    events = _remi_note_events(tokens, ticks_per_beat=ticks_per_beat)
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    scheduled = []
    target_tick = _target_ticks(duration_seconds, ticks_per_beat=ticks_per_beat, tempo=tempo)
    events = _fit_remi_events_to_duration(events, target_tick, ticks_per_beat=ticks_per_beat)
    last_tick = 0
    for event in events:
        start = int(event["tick"])
        duration = max(int(event["duration"]), 1)
        if target_tick is not None and start >= target_tick:
            continue
        if target_tick is not None:
            duration = min(duration, max(target_tick - start, 1))
        channel = _clamp(int(event["channel"]), 0, 15)
        note = _clamp(int(event["note"]), 0, 127)
        velocity = _clamp((int(event["velocity_bucket"]) * 8) + 4, 1, 127)
        scheduled.append(
            (
                start,
                0,
                Message(
                    "program_change",
                    channel=channel,
                    program=_clamp(int(event["program"]), 0, 127),
                    time=0,
                ),
            )
        )
        scheduled.append((start, 2, Message("note_on", channel=channel, note=note, velocity=velocity, time=0)))
        scheduled.append((start + duration, 1, Message("note_off", channel=channel, note=note, velocity=0, time=0)))
    if scheduled:
        for tick, _priority, message in sorted(scheduled, key=lambda item: (item[0], item[1])):
            message.time = max(int(tick) - last_tick, 0)
            track.append(message)
            last_tick = int(tick)
    else:
        note_length = int(second2tick(0.25, ticks_per_beat, tempo))
        total_notes = max(int((duration_seconds or 4) * 2), 1)
        for index in range(total_notes):
            note = 60 + (index % 12)
            track.append(Message("note_on", channel=0, note=note, velocity=72, time=0 if index == 0 else note_length))
            track.append(Message("note_off", channel=0, note=note, velocity=0, time=note_length))
            last_tick += note_length * 2
    _pad_track_to_duration(track, last_tick, duration_seconds, ticks_per_beat=ticks_per_beat, tempo=tempo)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_midi)
    return output_midi


def _remi_tokens_to_layered_midis(
    tokens: list[str],
    output_dir: Path,
    *,
    ticks_per_beat: int,
    tempo: int,
    duration_seconds: float | None = None,
) -> dict[str, str]:
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack, second2tick
    except ImportError as exc:
        raise RuntimeError("mido es necesario para exportar MIDI REMI por capas.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    midis: dict[str, MidiFile] = {}
    tracks: dict[str, MidiTrack] = {}
    last_ticks = {name: 0 for name in TRACK_NAMES}
    wrote_notes = {name: False for name in TRACK_NAMES}
    for name in TRACK_NAMES:
        midi = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        midi.tracks.append(track)
        track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
        midis[name] = midi
        tracks[name] = track
    target_tick = _target_ticks(duration_seconds, ticks_per_beat=ticks_per_beat, tempo=tempo)
    events = _fit_remi_events_to_duration(
        _remi_note_events(tokens, ticks_per_beat=ticks_per_beat),
        target_tick,
        ticks_per_beat=ticks_per_beat,
    )
    for event in events:
        layer = str(event.get("layer") or classify_midi_event_layer(
            channel=int(event["channel"]),
            note=int(event["note"]),
            program=int(event["program"]),
        ))
        if layer not in tracks:
            layer = "melody"
        start = int(event["tick"])
        duration = max(int(event["duration"]), 1)
        if target_tick is not None and start >= target_tick:
            continue
        if target_tick is not None:
            duration = min(duration, max(target_tick - start, 1))
        channel = _clamp(int(event["channel"]), 0, 15)
        note = _clamp(int(event["note"]), 0, 127)
        velocity = _clamp((int(event["velocity_bucket"]) * 8) + 4, 1, 127)
        delta = max(start - last_ticks[layer], 0)
        tracks[layer].append(
            Message(
                "program_change",
                channel=channel,
                program=_clamp(int(event["program"]), 0, 127),
                time=delta,
            )
        )
        tracks[layer].append(Message("note_on", channel=channel, note=note, velocity=velocity, time=0))
        tracks[layer].append(Message("note_off", channel=channel, note=note, velocity=0, time=duration))
        last_ticks[layer] = start + duration
        wrote_notes[layer] = True
    if not any(wrote_notes.values()):
        note_length = int(second2tick(0.25, ticks_per_beat, tempo))
        track = tracks["melody"]
        total_notes = max(int((duration_seconds or 4) * 2), 1)
        for index in range(total_notes):
            note = 60 + (index % 12)
            track.append(Message("note_on", channel=0, note=note, velocity=72, time=0 if index == 0 else note_length))
            track.append(Message("note_off", channel=0, note=note, velocity=0, time=note_length))
            last_ticks["melody"] += note_length * 2
    paths: dict[str, str] = {}
    for name, midi in midis.items():
        _pad_track_to_duration(
            tracks[name],
            last_ticks[name],
            duration_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        path = output_dir / f"{name}.mid"
        midi.save(path)
        paths[name] = str(path)
    return paths


def _remi_note_events(tokens: list[str], *, ticks_per_beat: int) -> list[dict[str, int | str]]:
    current_bar = 0
    current_position = 0
    current_program = {channel: 0 for channel in range(16)}
    current_layer = "melody"
    events: list[dict[str, int | str]] = []
    for token in tokens:
        parts = token.split(":")
        kind = parts[0]
        if kind == "bar" and len(parts) >= 2:
            current_bar = _safe_int(parts[1], current_bar)
        elif kind in {"position", "pos"} and len(parts) >= 2:
            current_position = _safe_int(parts[1], current_position)
        elif kind == "program" and len(parts) >= 3:
            channel = _clamp(_safe_int(parts[1], 0), 0, 15)
            current_program[channel] = _clamp(_safe_int(parts[2], 0), 0, 127)
        elif kind == "layer" and len(parts) >= 2:
            current_layer = parts[1] if parts[1] in TRACK_NAMES else "melody"
        elif kind == "note" and len(parts) >= 5:
            channel = _clamp(_safe_int(parts[1], 0), 0, 15)
            tick = ((current_bar * 16) + current_position) * max(ticks_per_beat // 4, 1)
            events.append(
                {
                    "tick": tick,
                    "channel": channel,
                    "note": _clamp(_safe_int(parts[2], 60), 0, 127),
                    "velocity_bucket": _clamp(_safe_int(parts[3], 8), 0, 15),
                    "duration": max(_safe_int(parts[4], ticks_per_beat // 2), 1),
                    "program": current_program.get(channel, 0),
                    "layer": current_layer,
                }
            )
    return sorted(events, key=lambda item: (int(item["tick"]), int(item["note"])))


def _first_ticks_per_beat(tokens: list[str]) -> int | None:
    for token in tokens:
        if token.startswith("ticks_per_beat:"):
            return _safe_int(token.split(":", 1)[1], 480)
    return None


def _first_tempo(tokens: list[str]) -> int | None:
    for token in tokens:
        if token.startswith("tempo:"):
            return _safe_int(token.split(":", 1)[1], 500000)
    return None


def _target_ticks(duration_seconds: float | None, *, ticks_per_beat: int, tempo: int) -> int | None:
    if duration_seconds is None or duration_seconds <= 0:
        return None
    seconds_per_beat = max(float(tempo), 1.0) / 1_000_000.0
    return max(int(round((duration_seconds / seconds_per_beat) * ticks_per_beat)), 1)


def _pad_track_to_duration(
    track,
    current_tick: int,
    duration_seconds: float | None,
    *,
    ticks_per_beat: int,
    tempo: int,
) -> None:
    target_tick = _target_ticks(duration_seconds, ticks_per_beat=ticks_per_beat, tempo=tempo)
    if target_tick is None:
        return
    delta = max(target_tick - int(current_tick), 0)
    if delta <= 0:
        return
    try:
        from mido import MetaMessage
    except ImportError as exc:
        raise RuntimeError("mido es necesario para ajustar duración MIDI.") from exc
    track.append(MetaMessage("end_of_track", time=delta))


def _fit_remi_events_to_duration(
    events: list[dict[str, int | str]],
    target_tick: int | None,
    *,
    ticks_per_beat: int,
) -> list[dict[str, int | str]]:
    if target_tick is None or not events:
        return events
    content_end = max(int(event["tick"]) + int(event["duration"]) for event in events)
    if content_end <= 0 or content_end >= int(target_tick * 0.9):
        return events

    bar_ticks = max(ticks_per_beat * 4, 1)
    cycle_ticks = max(((content_end + bar_ticks - 1) // bar_ticks) * bar_ticks, bar_ticks)
    fitted = list(events)
    offset = cycle_ticks
    while offset < target_tick:
        for event in events:
            copied = dict(event)
            copied["tick"] = int(event["tick"]) + offset
            if int(copied["tick"]) < target_tick:
                fitted.append(copied)
        offset += cycle_ticks
    return sorted(fitted, key=lambda item: (int(item["tick"]), int(item["note"])))


def _safe_int(value: str, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))
