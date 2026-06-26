from __future__ import annotations

from pathlib import Path
from typing import Any


LAYER_RANGES = {
    "bass": (24, 60),
    "melody": (48, 88),
    "harmony": (36, 88),
    "drums": (0, 127),
}


def cleanup_midi_layer(
    midi_path: Path,
    *,
    layer: str,
    quantize_grid: str = "1/16",
    min_note_ticks: int = 48,
    velocity_floor: int = 12,
    merge_gap_ticks: int = 24,
) -> dict[str, Any]:
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack
    except ImportError as exc:
        raise RuntimeError("La limpieza MIDI requiere mido.") from exc

    source = Path(midi_path).expanduser().resolve()
    midi = MidiFile(source)
    ticks_per_beat = midi.ticks_per_beat or 480
    grid_ticks = _grid_to_ticks(quantize_grid, ticks_per_beat)
    pitch_min, pitch_max = LAYER_RANGES.get(layer, (0, 127))
    is_drums = layer == "drums"

    notes: list[dict[str, int]] = []
    meta_events: list[MetaMessage] = []
    programs: list[Message] = []
    for track in midi.tracks:
        absolute_tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta:
                if message.type in {"set_tempo", "time_signature"}:
                    meta_events.append(message.copy(time=0))
                continue
            if message.type == "program_change":
                programs.append(message.copy(time=0))
                continue
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault((message.channel, message.note), []).append((absolute_tick, message.velocity))
                continue
            if message.type in {"note_off", "note_on"}:
                key = (message.channel, message.note)
                if key not in active or not active[key]:
                    continue
                start_tick, velocity = active[key].pop(0)
                if velocity < velocity_floor:
                    continue
                if not is_drums and not (pitch_min <= message.note <= pitch_max):
                    continue
                end_tick = max(absolute_tick, start_tick + 1)
                q_start = _quantize_tick(start_tick, grid_ticks)
                q_end = max(_quantize_tick(end_tick, grid_ticks), q_start + min_note_ticks)
                if q_end - q_start < min_note_ticks:
                    continue
                notes.append(
                    {
                        "channel": 9 if is_drums else min(message.channel, 8),
                        "note": message.note,
                        "velocity": max(1, min(127, velocity)),
                        "start": max(q_start, 0),
                        "end": max(q_end, 1),
                    }
                )

    notes = _merge_repeated_notes(notes, merge_gap_ticks=merge_gap_ticks)
    output = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    output.tracks.append(track)
    track.append(MetaMessage("track_name", name=layer.title(), time=0))
    for meta in _unique_meta_events(meta_events):
        track.append(meta.copy(time=0))
    if not is_drums:
        program = _first_program(programs)
        track.append(program if program else Message("program_change", channel=0, program=0, time=0))
    else:
        track.append(Message("program_change", channel=9, program=0, time=0))

    events: list[tuple[int, Message]] = []
    for note in notes:
        events.append(
            (
                note["start"],
                Message(
                    "note_on",
                    channel=note["channel"],
                    note=note["note"],
                    velocity=note["velocity"],
                    time=0,
                ),
            )
        )
        events.append(
            (
                note["end"],
                Message("note_off", channel=note["channel"], note=note["note"], velocity=0, time=0),
            )
        )

    previous_tick = 0
    for tick, message in sorted(events, key=lambda item: (item[0], item[1].type != "note_off")):
        message.time = max(tick - previous_tick, 0)
        track.append(message)
        previous_tick = tick

    output.save(source)
    duration_ticks = max((note["end"] for note in notes), default=0)
    return {
        "path": str(source),
        "layer": layer,
        "note_count": len(notes),
        "duration_ticks": duration_ticks,
        "pitch_range": [pitch_min, pitch_max],
        "quantize_grid": quantize_grid,
        "grid_ticks": grid_ticks,
        "quality_score": _quality_score(notes, duration_ticks),
        "valid": bool(notes),
    }


def _grid_to_ticks(grid: str, ticks_per_beat: int) -> int:
    values = {
        "1/8": ticks_per_beat // 2,
        "1/16": ticks_per_beat // 4,
        "1/32": ticks_per_beat // 8,
    }
    return max(values.get(grid, ticks_per_beat // 4), 1)


def _quantize_tick(tick: int, grid_ticks: int) -> int:
    return int(round(tick / grid_ticks) * grid_ticks)


def _merge_repeated_notes(notes: list[dict[str, int]], *, merge_gap_ticks: int) -> list[dict[str, int]]:
    merged: list[dict[str, int]] = []
    for note in sorted(notes, key=lambda item: (item["channel"], item["note"], item["start"])):
        if (
            merged
            and merged[-1]["channel"] == note["channel"]
            and merged[-1]["note"] == note["note"]
            and note["start"] - merged[-1]["end"] <= merge_gap_ticks
        ):
            merged[-1]["end"] = max(merged[-1]["end"], note["end"])
            merged[-1]["velocity"] = max(merged[-1]["velocity"], note["velocity"])
        else:
            merged.append(note.copy())
    return sorted(merged, key=lambda item: item["start"])


def _unique_meta_events(events: list[Any]) -> list[Any]:
    unique = []
    seen = set()
    for event in events:
        key = event.type
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def _first_program(programs: list[Any]) -> Any | None:
    for program in programs:
        if getattr(program, "channel", 0) != 9:
            return program.copy(time=0)
    return None


def _quality_score(notes: list[dict[str, int]], duration_ticks: int) -> float:
    if not notes or duration_ticks <= 0:
        return 0.0
    unique_pitches = len({note["note"] for note in notes})
    density = min(len(notes) / max(duration_ticks / 480, 1), 8) / 8
    diversity = min(unique_pitches / 12, 1)
    return round((density * 0.45) + (diversity * 0.55), 4)
