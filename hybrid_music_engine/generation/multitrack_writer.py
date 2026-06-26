from __future__ import annotations

from pathlib import Path


TRACK_NAMES = ["drums", "bass", "harmony", "melody"]

BASS_PROGRAMS = set(range(32, 40))


def expected_multitrack_paths(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {name: str(output_dir / f"{name}.mid") for name in TRACK_NAMES}


def classify_midi_event_layer(
    *,
    channel: int,
    note: int | None = None,
    program: int | None = None,
) -> str:
    if channel == 9:
        return "drums"
    if program in BASS_PROGRAMS:
        return "bass"
    if note is not None and note < 55:
        return "bass"
    if note is not None and note < 72:
        return "harmony"
    return "melody"
