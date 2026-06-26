from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def transcribe_melodic_basic_pitch(
    audio_path: Path,
    output_midi: Path,
    *,
    minimum_note_length: float | None = None,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise RuntimeError(f"Audio para transcripción no encontrado: {audio_path}")
    output_midi.parent.mkdir(parents=True, exist_ok=True)

    try:
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError as exc:
        return _transcribe_melodic_librosa(
            audio_path,
            output_midi,
            minimum_note_length=minimum_note_length,
            import_error=exc,
        )

    temp_dir = output_midi.parent / f".basic_pitch_{output_midi.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if minimum_note_length is not None:
        kwargs["minimum_note_length"] = minimum_note_length
    if onset_threshold is not None:
        kwargs["onset_threshold"] = onset_threshold
    if frame_threshold is not None:
        kwargs["frame_threshold"] = frame_threshold

    predict_and_save(
        [str(audio_path)],
        str(temp_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        **kwargs,
    )

    generated = sorted(temp_dir.glob("*.mid")) + sorted(temp_dir.glob("*.midi"))
    if not generated:
        raise RuntimeError("Basic Pitch no generó un archivo MIDI.")
    shutil.move(str(generated[0]), output_midi)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {
        "source": str(audio_path),
        "midi": str(output_midi),
        "engine": "basic-pitch",
        "minimum_note_length": minimum_note_length,
        "onset_threshold": onset_threshold,
        "frame_threshold": frame_threshold,
    }


def _transcribe_melodic_librosa(
    audio_path: Path,
    output_midi: Path,
    *,
    minimum_note_length: float | None,
    import_error: ImportError,
) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
    except ImportError as exc:
        raise RuntimeError(
            "Basic Pitch no está disponible para este entorno y tampoco se pudo usar "
            "el fallback con librosa. Para Basic Pitch usa un venv con Python 3.11."
        ) from exc

    sample_rate = 22050
    hop_length = 512
    min_duration = minimum_note_length if minimum_note_length is not None else 0.08
    audio, loaded_sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise RuntimeError(f"El audio melódico está vacío: {audio_path}") from import_error

    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=loaded_sr,
        hop_length=hop_length,
    )
    midi_notes = np.rint(librosa.hz_to_midi(f0)).astype(float)
    frames = librosa.frames_to_time(np.arange(len(midi_notes)), sr=loaded_sr, hop_length=hop_length)

    segments: list[tuple[int, float, float]] = []
    active_note: int | None = None
    start_time = 0.0
    for index, note_value in enumerate(midi_notes):
        note = int(note_value) if bool(voiced[index]) and np.isfinite(note_value) else None
        current_time = float(frames[index])
        if note == active_note:
            continue
        if active_note is not None and current_time - start_time >= min_duration:
            segments.append((active_note, start_time, current_time))
        active_note = note
        start_time = current_time

    end_time = float(len(audio) / loaded_sr)
    if active_note is not None and end_time - start_time >= min_duration:
        segments.append((active_note, start_time, end_time))
    if not segments:
        raise RuntimeError(
            "No se detectaron notas melódicas con el fallback de librosa. "
            "Para mejor transcripción instala Basic Pitch en Python 3.11."
        ) from import_error

    tempo = bpm2tempo(120)
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(Message("program_change", program=0, channel=0, time=0))

    events: list[tuple[int, Message]] = []
    for note, start, end in segments:
        start_tick = int(second2tick(start, midi.ticks_per_beat, tempo))
        end_tick = max(int(second2tick(end, midi.ticks_per_beat, tempo)), start_tick + 1)
        events.append((start_tick, Message("note_on", note=note, velocity=80, channel=0, time=0)))
        events.append((end_tick, Message("note_off", note=note, velocity=0, channel=0, time=0)))

    last_tick = 0
    for tick, message in sorted(events, key=lambda item: item[0]):
        message.time = max(tick - last_tick, 0)
        track.append(message)
        last_tick = tick
    midi.save(output_midi)

    return {
        "source": str(audio_path),
        "midi": str(output_midi),
        "engine": "librosa-pyin-fallback",
        "minimum_note_length": minimum_note_length,
        "warning": "Basic Pitch no está disponible en este Python; se usó transcripción monofónica aproximada.",
    }
