from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def analyze_midi_quality(midi_path: Path) -> dict[str, Any]:
    try:
        from mido import MidiFile, tick2second
    except ImportError as exc:
        raise RuntimeError("mido es necesario para calcular métricas MIDI.") from exc

    path = Path(midi_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"MIDI no encontrado: {path}")

    midi = MidiFile(path)
    tempo = 500000
    notes: list[dict[str, Any]] = []
    programs: set[int] = set()
    channels: set[int] = set()
    rhythmic_ticks: list[int] = []
    max_seconds = 0.0

    for track in midi.tracks:
        absolute_ticks = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for message in track:
            absolute_ticks += int(message.time)
            if message.type == "set_tempo":
                tempo = int(message.tempo)
            seconds = float(tick2second(absolute_ticks, midi.ticks_per_beat, tempo))
            max_seconds = max(max_seconds, seconds)
            if hasattr(message, "channel"):
                channels.add(int(message.channel))
            if message.type == "program_change":
                programs.add(int(message.program))
            if message.type == "note_on" and message.velocity > 0:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                active.setdefault(key, []).append((absolute_ticks, int(message.velocity)))
                rhythmic_ticks.append(absolute_ticks)
            elif message.type in {"note_off", "note_on"}:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                starts = active.get(key)
                if not starts:
                    continue
                start_tick, velocity = starts.pop(0)
                start = float(tick2second(start_tick, midi.ticks_per_beat, tempo))
                duration = max(seconds - start, 0.0)
                notes.append(
                    {
                        "note": key[1],
                        "channel": key[0],
                        "start": start,
                        "duration": duration,
                        "velocity": velocity,
                    }
                )
                max_seconds = max(max_seconds, seconds)

    note_count = len(notes)
    pitch_values = [int(note["note"]) for note in notes]
    durations = [float(note["duration"]) for note in notes if float(note["duration"]) > 0]
    pitch_classes = {pitch % 12 for pitch in pitch_values}
    unique_pitches = set(pitch_values)
    duration_seconds = max(max_seconds, sum(durations), 0.0)
    density = note_count / duration_seconds if duration_seconds > 0 else 0.0
    rhythm_diversity = len(Counter(_tick_bin(value, midi.ticks_per_beat) for value in rhythmic_ticks))
    repeated_ratio = _max_repeated_pitch_ratio(pitch_values)
    silence_risk = note_count == 0 or density < 0.1
    validity_score = _score(
        note_count=note_count,
        duration_seconds=duration_seconds,
        pitch_diversity=len(unique_pitches),
        pitch_class_diversity=len(pitch_classes),
        rhythm_diversity=rhythm_diversity,
        repeated_ratio=repeated_ratio,
        silence_risk=silence_risk,
    )

    return {
        "source_midi": str(path),
        "valid_midi": True,
        "note_count": note_count,
        "duration_seconds": round(duration_seconds, 4),
        "note_density_per_second": round(density, 4),
        "unique_pitches": len(unique_pitches),
        "pitch_class_diversity": len(pitch_classes),
        "rhythm_diversity": rhythm_diversity,
        "program_count": len(programs),
        "programs": sorted(programs),
        "channels": sorted(channels),
        "repeated_pitch_ratio": round(repeated_ratio, 4),
        "silence_risk": silence_risk,
        "quality_score": validity_score,
    }


def _tick_bin(value: int, ticks_per_beat: int) -> int:
    step = max(ticks_per_beat // 4, 1)
    return int(value // step)


def _max_repeated_pitch_ratio(pitches: list[int]) -> float:
    if not pitches:
        return 1.0
    counts = Counter(pitches)
    return max(counts.values()) / len(pitches)


def _score(
    *,
    note_count: int,
    duration_seconds: float,
    pitch_diversity: int,
    pitch_class_diversity: int,
    rhythm_diversity: int,
    repeated_ratio: float,
    silence_risk: bool,
) -> float:
    if silence_risk:
        return 0.0
    score = 0.0
    score += min(note_count / 80.0, 1.0) * 0.22
    score += min(duration_seconds / 30.0, 1.0) * 0.14
    score += min(pitch_diversity / 16.0, 1.0) * 0.22
    score += min(pitch_class_diversity / 10.0, 1.0) * 0.18
    score += min(rhythm_diversity / 24.0, 1.0) * 0.16
    score += max(1.0 - repeated_ratio, 0.0) * 0.08
    return round(min(score, 1.0), 4)
