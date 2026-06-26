from __future__ import annotations

from pathlib import Path


GM_DRUM_MAP = {
    "kick": 36,
    "snare": 38,
    "closed_hat": 42,
}


def _as_float(value: object, fallback: float) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _classify_drum_hit(low_ratio: float, high_ratio: float) -> str:
    if low_ratio >= 0.42:
        return "kick"
    if high_ratio >= 0.38:
        return "closed_hat"
    return "snare"


def transcribe_drums_onsets(
    audio_path: Path,
    output_midi: Path,
    *,
    bpm: float | None = None,
    onset_delta: float = 0.07,
    onset_wait: float = 0.03,
    note_length: float = 0.08,
    velocity_floor: int = 45,
    velocity_ceiling: int = 115,
) -> dict:
    try:
        import librosa
        import numpy as np
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
    except ImportError as exc:
        raise RuntimeError(
            "La transcripción percusiva requiere librosa, numpy y mido. "
            "Instala el proyecto con: python -m pip install -e ."
        ) from exc

    source = Path(audio_path).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"Audio percusivo no encontrado: {source}")

    y, sample_rate = librosa.load(source, sr=None, mono=True)
    if len(y) == 0:
        raise RuntimeError(f"El audio percusivo está vacío: {source}")

    hop_length = 512
    n_fft = 2048
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        delta=onset_delta,
        wait=max(int(onset_wait * sample_rate / hop_length), 1),
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)

    if bpm is None:
        estimated_bpm, _beats = librosa.beat.beat_track(
            y=y,
            sr=sample_rate,
            hop_length=hop_length,
        )
        bpm = _as_float(estimated_bpm, 120.0)
    bpm = max(_as_float(bpm, 120.0), 20.0)

    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    low_mask = freqs < 180
    high_mask = freqs >= 4000

    max_strength = float(np.max(onset_envelope)) if len(onset_envelope) else 0.0
    tempo = bpm2tempo(bpm)
    ticks_per_beat = 480
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("track_name", name="Drums", time=0))
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(Message("program_change", channel=9, program=0, time=0))

    absolute_events: list[tuple[int, Message]] = []
    drum_counts = {name: 0 for name in GM_DRUM_MAP}
    for frame, onset_time in zip(onset_frames, onset_times, strict=False):
        column = spectrum[:, min(int(frame), spectrum.shape[1] - 1)]
        total_energy = float(np.sum(column)) + 1e-9
        low_ratio = float(np.sum(column[low_mask]) / total_energy)
        high_ratio = float(np.sum(column[high_mask]) / total_energy)
        drum_name = _classify_drum_hit(low_ratio, high_ratio)
        note = GM_DRUM_MAP[drum_name]
        strength = float(onset_envelope[min(int(frame), len(onset_envelope) - 1)])
        if max_strength > 0:
            normalized = max(0.0, min(strength / max_strength, 1.0))
        else:
            normalized = 0.65
        velocity = int(velocity_floor + normalized * (velocity_ceiling - velocity_floor))
        velocity = max(1, min(127, velocity))
        start_tick = int(second2tick(float(onset_time), ticks_per_beat, tempo))
        end_tick = int(second2tick(float(onset_time + note_length), ticks_per_beat, tempo))
        absolute_events.append(
            (start_tick, Message("note_on", channel=9, note=note, velocity=velocity, time=0))
        )
        absolute_events.append(
            (max(end_tick, start_tick + 1), Message("note_off", channel=9, note=note, velocity=0, time=0))
        )
        drum_counts[drum_name] += 1

    previous_tick = 0
    for absolute_tick, message in sorted(absolute_events, key=lambda item: item[0]):
        message.time = max(absolute_tick - previous_tick, 0)
        track.append(message)
        previous_tick = absolute_tick

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_midi)
    return {
        "source": str(source),
        "midi": str(output_midi),
        "engine": "librosa-onsets",
        "sample_rate": int(sample_rate),
        "bpm": bpm,
        "note_count": len(onset_frames),
        "drum_counts": drum_counts,
        "gm_drum_map": GM_DRUM_MAP,
        "parameters": {
            "onset_delta": onset_delta,
            "onset_wait": onset_wait,
            "note_length": note_length,
            "velocity_floor": velocity_floor,
            "velocity_ceiling": velocity_ceiling,
        },
    }
