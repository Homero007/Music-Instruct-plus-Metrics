from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.audio.loader import analyze_audio, _load_audio
from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.storage.manifest import ProjectManifest, project_path


def extract_global_audio_features(audio_path: Path, config: EngineConfig) -> dict:
    audio, sample_rate = _load_audio(audio_path, target_sample_rate=config.default_sample_rate)
    summary = analyze_audio(audio, sample_rate)
    mono = audio.mean(axis=0)
    if mono.size == 0:
        return summary

    try:
        import librosa

        spectral_centroid = librosa.feature.spectral_centroid(y=mono, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=mono)
        onset_envelope = librosa.onset.onset_strength(y=mono, sr=sample_rate)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            units="frames",
        )
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sample_rate)
        pitch_class_profile = _normalize_histogram(np.mean(chroma, axis=1))
        key_payload = _estimate_key_mode(pitch_class_profile)
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
        onset_strength = np.asarray(onset_envelope, dtype=float)
        summary.update(
            {
                "spectral_centroid_mean": _round_float(np.mean(spectral_centroid)),
                "spectral_bandwidth_mean": _round_float(np.mean(spectral_bandwidth)),
                "spectral_rolloff_mean": _round_float(np.mean(spectral_rolloff)),
                "zero_crossing_rate_mean": _round_float(np.mean(zero_crossing_rate), digits=8),
                "onset_count": int(len(onset_frames)),
                "onset_density_per_second": _safe_density(len(onset_frames), summary["duration_seconds"]),
                "groove_vector": _groove_vector(onset_times, float(summary["duration_seconds"])),
                "structure": _audio_structure(onset_strength, float(summary["duration_seconds"])),
                "pitch_class_profile": pitch_class_profile,
                "estimated_key_index": key_payload["key_index"],
                "estimated_key": key_payload["key"],
                "estimated_mode": key_payload["mode"],
                "key_confidence": key_payload["confidence"],
            }
        )
    except (ImportError, ValueError, RuntimeError, TypeError):
        summary.update(
            {
                "onset_count": None,
                "onset_density_per_second": None,
                "pitch_class_profile": [],
                "estimated_key_index": None,
                "estimated_key": None,
            }
        )
    return summary


def extract_midi_features(midi_path: Path) -> dict[str, Any]:
    try:
        from mido import MidiFile, tick2second
    except ImportError as exc:
        raise RuntimeError("mido es necesario para extraer features MIDI.") from exc

    path = Path(midi_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"MIDI no encontrado: {path}")

    midi = MidiFile(path)
    tempo = 500000
    note_starts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[dict[str, Any]] = []
    programs: set[int] = set()
    channels: set[int] = set()
    drum_notes: dict[int, int] = {}
    max_tick = 0

    for track in midi.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            max_tick = max(max_tick, absolute_tick)
            if message.type == "set_tempo":
                tempo = int(message.tempo)
            if hasattr(message, "channel"):
                channels.add(int(message.channel))
            if message.type == "program_change":
                programs.add(int(message.program))
            if message.type == "note_on" and message.velocity > 0:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                note_starts.setdefault(key, []).append((absolute_tick, int(message.velocity)))
            elif message.type in {"note_off", "note_on"}:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                starts = note_starts.get(key)
                if not starts:
                    continue
                start_tick, velocity = starts.pop(0)
                end_tick = max(absolute_tick, start_tick + 1)
                channel, pitch = key
                start_second = tick2second(start_tick, midi.ticks_per_beat, tempo)
                end_second = tick2second(end_tick, midi.ticks_per_beat, tempo)
                notes.append(
                    {
                        "pitch": pitch,
                        "velocity": velocity,
                        "channel": channel,
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "start_second": float(start_second),
                        "duration_seconds": max(float(end_second - start_second), 0.0),
                    }
                )
                if channel == 9:
                    drum_notes[pitch] = drum_notes.get(pitch, 0) + 1

    duration_seconds = float(tick2second(max_tick, midi.ticks_per_beat, tempo)) if max_tick else 0.0
    pitches = [note["pitch"] for note in notes]
    velocities = [note["velocity"] for note in notes]
    durations = [note["duration_seconds"] for note in notes]
    starts = sorted(note["start_second"] for note in notes)
    intervals = [round(starts[i] - starts[i - 1], 4) for i in range(1, len(starts))]
    pitch_classes = [pitch % 12 for pitch in pitches]
    pitch_histogram = _histogram(pitch_classes, bins=12)
    pitch_profile = _normalize_histogram(np.asarray(pitch_histogram, dtype=float))
    key_payload = _estimate_key_mode(pitch_profile)
    rhythmic_payload = _rhythmic_features(notes, midi.ticks_per_beat)
    harmonic_payload = _harmonic_features(notes, midi.ticks_per_beat)

    return {
        "path": str(path),
        "ticks_per_beat": int(midi.ticks_per_beat),
        "duration_seconds": _round_float(duration_seconds),
        "note_count": len(notes),
        "note_density_per_second": _safe_density(len(notes), duration_seconds),
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "pitch_mean": _round_float(np.mean(pitches)) if pitches else None,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else None,
        "pitch_diversity": len(set(pitches)),
        "pitch_class_diversity": len(set(pitch_classes)),
        "pitch_class_histogram": pitch_histogram,
        "pitch_class_profile": pitch_profile,
        "estimated_key_index": key_payload["key_index"],
        "estimated_key": key_payload["key"],
        "estimated_mode": key_payload["mode"],
        "key_confidence": key_payload["confidence"],
        "chord_hints": harmonic_payload["chord_hints"],
        "harmonic_tension": harmonic_payload["harmonic_tension"],
        "velocity_mean": _round_float(np.mean(velocities)) if velocities else None,
        "velocity_std": _round_float(np.std(velocities)) if velocities else None,
        "duration_mean": _round_float(np.mean(durations)) if durations else None,
        "duration_std": _round_float(np.std(durations)) if durations else None,
        "rhythmic_diversity": len(set(intervals)),
        "inter_onset_mean": _round_float(np.mean(intervals)) if intervals else None,
        "swing_ratio": rhythmic_payload["swing_ratio"],
        "syncopation_score": rhythmic_payload["syncopation_score"],
        "groove_vector": rhythmic_payload["groove_vector"],
        "structure": _midi_structure(notes, duration_seconds),
        "channels": sorted(channels),
        "programs": sorted(programs),
        "is_drum_track": 9 in channels,
        "drum_note_counts": {str(key): value for key, value in sorted(drum_notes.items())},
    }


def extract_project_features(
    manifest: ProjectManifest,
    config: EngineConfig,
    *,
    include_audio: bool = True,
    include_midis: bool = True,
) -> dict[str, Any]:
    project_dir = project_path(config, manifest.project_id)
    audio_features: dict[str, Any] = {}
    midi_features: dict[str, Any] = {}

    normalized = manifest.source.get("normalized")
    if include_audio and normalized:
        audio_features["normalized"] = extract_global_audio_features(Path(str(normalized)), config)

    if include_midis:
        for label, midi_path in _collect_midi_paths(manifest, project_dir).items():
            midi_features[label] = extract_midi_features(midi_path)

    summary = _summarize_features(audio_features, midi_features)
    payload = {
        "project_id": manifest.project_id,
        "audio": audio_features,
        "midi": midi_features,
        "summary": summary,
    }
    output_path = project_dir / "features" / "features.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "path": str(output_path)}


def _collect_midi_paths(manifest: ProjectManifest, project_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    drums = manifest.midis.get("drums", {})
    if isinstance(drums, dict) and drums.get("midi"):
        paths["drums"] = Path(str(drums["midi"]))

    melodic = manifest.midis.get("melodic", {})
    if isinstance(melodic, dict):
        for name, payload in melodic.items():
            if isinstance(payload, dict) and payload.get("midi"):
                paths[f"melodic.{name}"] = Path(str(payload["midi"]))

    for path in sorted((project_dir / "midis").glob("*.mid")):
        paths.setdefault(path.stem, path)
    for path in sorted((project_dir / "midis").glob("*.midi")):
        paths.setdefault(path.stem, path)
    return paths


def _summarize_features(audio: dict[str, Any], midi: dict[str, Any]) -> dict[str, Any]:
    midi_note_counts = {
        label: features.get("note_count", 0)
        for label, features in midi.items()
    }
    total_notes = int(sum(midi_note_counts.values()))
    durations = [
        float(features["duration_seconds"])
        for features in midi.values()
        if features.get("duration_seconds") is not None
    ]
    return {
        "audio_available": bool(audio),
        "midi_layers": sorted(midi.keys()),
        "midi_layer_count": len(midi),
        "midi_note_counts": midi_note_counts,
        "total_midi_notes": total_notes,
        "max_midi_duration_seconds": _round_float(max(durations)) if durations else 0.0,
        "drums_available": "drums" in midi,
        "melodic_layers": sorted(label for label in midi if label.startswith("melodic.")),
    }


def _round_float(value: object, digits: int = 4) -> float:
    return round(float(value), digits)


def _safe_density(count: int, duration_seconds: float | int | None) -> float:
    duration = float(duration_seconds or 0.0)
    if duration <= 1e-9:
        return 0.0
    return round(float(count) / duration, 4)


def _histogram(values: list[int], bins: int) -> list[int]:
    histogram = [0 for _ in range(bins)]
    for value in values:
        if 0 <= value < bins:
            histogram[value] += 1
    return histogram


def _normalize_histogram(values: np.ndarray) -> list[float]:
    total = float(np.sum(values))
    if total <= 1e-9:
        return [0.0 for _ in values]
    return [round(float(value / total), 6) for value in values]


def _pitch_class_name(index: int | None) -> str | None:
    if index is None:
        return None
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[index % 12]


def _estimate_key_mode(profile: list[float]) -> dict[str, Any]:
    if not profile or len(profile) < 12:
        return {"key_index": None, "key": None, "mode": None, "confidence": 0.0}
    values = np.asarray(profile[:12], dtype=float)
    if float(values.sum()) <= 1e-9:
        return {"key_index": None, "key": None, "mode": None, "confidence": 0.0}
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores: list[tuple[float, int, str]] = []
    for index in range(12):
        scores.append((float(np.dot(values, np.roll(major, index))), index, "major"))
        scores.append((float(np.dot(values, np.roll(minor, index))), index, "minor"))
    scores.sort(reverse=True)
    best, key_index, mode = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    confidence = (best - second) / max(abs(best), 1e-9)
    return {
        "key_index": int(key_index),
        "key": _pitch_class_name(key_index),
        "mode": mode,
        "confidence": round(float(max(confidence, 0.0)), 4),
    }


def _rhythmic_features(notes: list[dict[str, Any]], ticks_per_beat: int) -> dict[str, Any]:
    if not notes:
        return {"swing_ratio": None, "syncopation_score": 0.0, "groove_vector": [0.0] * 16}
    starts = sorted(int(note["start_tick"]) for note in notes)
    sixteenth = max(ticks_per_beat // 4, 1)
    groove = [0 for _ in range(16)]
    offbeat = 0
    weak_velocity = 0
    strong_velocity = 0
    for note in notes:
        slot = int(round(int(note["start_tick"]) / sixteenth)) % 16
        groove[slot] += 1
        velocity = int(note.get("velocity", 0))
        if slot % 4 in {1, 3}:
            offbeat += 1
            weak_velocity += velocity
        else:
            strong_velocity += velocity
    short_long_pairs = []
    for first, second in zip(starts[::2], starts[1::2], strict=False):
        delta = max(second - first, 1)
        short_long_pairs.append(delta)
    swing_ratio = None
    if len(short_long_pairs) >= 2:
        odd = np.mean(short_long_pairs[::2])
        even = np.mean(short_long_pairs[1::2])
        swing_ratio = _round_float(max(odd, even) / max(min(odd, even), 1), 4)
    total = max(sum(groove), 1)
    syncopation = (offbeat / total) * 0.7
    if weak_velocity + strong_velocity > 0:
        syncopation += (weak_velocity / max(weak_velocity + strong_velocity, 1)) * 0.3
    return {
        "swing_ratio": swing_ratio,
        "syncopation_score": round(float(min(syncopation, 1.0)), 4),
        "groove_vector": [round(value / total, 4) for value in groove],
    }


def _groove_vector(onset_times: np.ndarray, duration_seconds: float) -> list[float]:
    if onset_times.size == 0 or duration_seconds <= 0:
        return [0.0] * 16
    slots = [0 for _ in range(16)]
    for onset in onset_times:
        position = (float(onset) / duration_seconds) % 1.0
        slot = min(int(position * 16), 15)
        slots[slot] += 1
    total = max(sum(slots), 1)
    return [round(value / total, 4) for value in slots]


def _harmonic_features(notes: list[dict[str, Any]], ticks_per_beat: int) -> dict[str, Any]:
    if not notes:
        return {"chord_hints": [], "harmonic_tension": 0.0}
    window = max(ticks_per_beat * 4, 1)
    buckets: dict[int, list[int]] = {}
    for note in notes:
        bucket = int(note["start_tick"]) // window
        buckets.setdefault(bucket, []).append(int(note["pitch"]) % 12)
    hints = []
    roots = []
    previous_profile: set[int] | None = None
    tension_values = []
    for bucket, pitch_classes in sorted(buckets.items()):
        counts = _histogram(pitch_classes, 12)
        root = int(np.argmax(counts)) if sum(counts) else 0
        profile = {index for index, count in enumerate(counts) if count}
        quality = _chord_quality(profile, root)
        hints.append({"window": bucket, "root": _pitch_class_name(root), "quality": quality})
        roots.append(root)
        if previous_profile is not None:
            union = len(previous_profile | profile) or 1
            tension_values.append(1.0 - (len(previous_profile & profile) / union))
        previous_profile = profile
    return {
        "chord_hints": hints[:32],
        "harmonic_tension": _round_float(np.mean(tension_values), 4) if tension_values else 0.0,
    }


def _chord_quality(profile: set[int], root: int) -> str:
    relative = {(value - root) % 12 for value in profile}
    if {0, 4, 7}.issubset(relative):
        return "major"
    if {0, 3, 7}.issubset(relative):
        return "minor"
    if {0, 3, 6}.issubset(relative):
        return "diminished"
    if {0, 4, 8}.issubset(relative):
        return "augmented"
    return "unknown"


def _audio_structure(onset_strength: np.ndarray, duration_seconds: float) -> list[dict[str, Any]]:
    if onset_strength.size == 0 or duration_seconds <= 0:
        return []
    chunks = np.array_split(onset_strength, min(4, max(1, onset_strength.size)))
    labels = ["intro", "body", "drop", "outro"][: len(chunks)]
    step = duration_seconds / len(chunks)
    return [
        {
            "label": label,
            "start_seconds": round(index * step, 4),
            "end_seconds": round((index + 1) * step, 4),
            "energy": _round_float(np.mean(chunk), 4),
        }
        for index, (label, chunk) in enumerate(zip(labels, chunks, strict=False))
    ]


def _midi_structure(notes: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    if not notes or duration_seconds <= 0:
        return []
    labels = ["intro", "body", "drop", "outro"]
    step = duration_seconds / len(labels)
    sections = []
    for index, label in enumerate(labels):
        start = index * step
        end = (index + 1) * step
        section_notes = [note for note in notes if start <= float(note["start_second"]) < end]
        sections.append(
            {
                "label": label,
                "start_seconds": round(start, 4),
                "end_seconds": round(end, 4),
                "note_density": _safe_density(len(section_notes), step),
                "mean_velocity": _round_float(np.mean([n["velocity"] for n in section_notes]), 4)
                if section_notes
                else 0.0,
            }
        )
    return sections
