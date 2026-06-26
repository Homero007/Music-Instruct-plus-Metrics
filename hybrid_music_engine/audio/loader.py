from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


def import_and_normalize_audio(
    source_path: Path,
    destination_dir: Path,
    config: EngineConfig,
) -> dict[str, Any]:
    if not source_path.exists() or not source_path.is_file():
        raise RuntimeError(f"Audio no encontrado: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise RuntimeError(f"Formato de audio no soportado: {source_path.suffix}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    original_path = destination_dir / f"original{source_path.suffix.lower()}"
    if source_path.resolve() != original_path.resolve():
        shutil.copy2(source_path, original_path)

    audio, sample_rate = _load_audio(original_path, target_sample_rate=config.default_sample_rate)
    audio = _ensure_channels(audio, config.default_channels)
    normalized = normalize_peak(audio)
    normalized_path = destination_dir / "normalized.wav"
    _write_audio(normalized_path, normalized, config.default_sample_rate)

    features = analyze_audio(normalized, config.default_sample_rate)
    return {
        "original": str(original_path),
        "normalized": str(normalized_path),
        "analysis": features,
    }


def _load_audio(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa es necesario para cargar y remuestrear audio.") from exc

    audio, sample_rate = librosa.load(
        path,
        sr=target_sample_rate,
        mono=False,
        dtype=np.float32,
    )
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    return audio, sample_rate


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile es necesario para escribir WAV.") from exc

    sf.write(path, audio.T, sample_rate, subtype="PCM_24")


def _ensure_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return audio.mean(axis=0, keepdims=True)
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1 and channels == 2:
        return np.repeat(audio, 2, axis=0)
    return audio[:channels]


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return audio.astype(np.float32)
    return (audio * (target_peak / peak)).astype(np.float32)


def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mono = audio.mean(axis=0)
    duration = float(mono.size / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0

    tempo: float | None = None
    beat_count = 0
    try:
        import librosa

        tempo_value, beat_frames = librosa.beat.beat_track(y=mono, sr=sample_rate)
        tempo = float(np.asarray(tempo_value).reshape(-1)[0])
        beat_count = int(len(beat_frames))
    except (ImportError, ValueError, RuntimeError, TypeError):
        tempo = None

    return {
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "duration_seconds": round(duration, 4),
        "rms": round(rms, 8),
        "peak": round(peak, 8),
        "tempo_bpm": round(tempo, 4) if tempo is not None else None,
        "beat_count": beat_count,
    }
