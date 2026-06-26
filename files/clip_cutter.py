#!/usr/bin/env python3
"""
clip_cutter.py — Corte de clips en STREAMING (sin cargar la pista completa).

Reemplazo directo del fragmento de `jamendo.prepare_jamendo_clips` que hoy hace
`sf.read(source)` del archivo ENTERO y luego rebana en memoria. Para una pista de
~9 min estéreo eso reserva ~197 MiB de una sola vez (el error
`Unable to allocate ... shape (25785408, 2) float32`). Aquí leemos SOLO la ventana
de cada clip con `sf.SoundFile.seek()/read()`, así el pico de memoria es del
tamaño de UN clip, no de la pista completa.

Comportamiento preservado
-------------------------
- Mismo esquema de `entries` y mismas razones de rechazo que el código original.
- Si el sample rate del archivo != objetivo, o soundfile no puede abrirlo (p.ej.
  ciertos MP3), cae a librosa leyendo POR CLIP con offset/duration (tampoco carga
  todo el archivo).
- Escribe cada clip con `sf.write(..., subtype="PCM_24")`, igual que antes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


def iter_windows(
    total_samples: int,
    clip_samples: int,
    hop_samples: int,
    min_samples: int,
    max_clips: int | None,
) -> Iterator[tuple[int, int]]:
    """Genera (start, end) por clip, idéntico al recorrido del loop original."""
    start = 0
    count = 0
    while start < total_samples:
        if max_clips is not None and count >= max_clips:
            return
        end = min(start + clip_samples, total_samples)
        if end - start < min_samples:
            return
        yield start, end
        count += 1
        start += hop_samples


def _entry(
    *, clip_id, track_id, genre, clip_index, source, clip_path,
    start_sample, end_sample, loaded_sr, track,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "track_id": track_id,
        "genre": genre,
        "clip_index": clip_index,
        "source_audio": str(source),
        "clip_path": str(clip_path),
        "start_seconds": round(start_sample / loaded_sr, 4),
        "duration_seconds": round((end_sample - start_sample) / loaded_sr, 4),
        "sample_rate": int(loaded_sr),
        "source_track": {
            "name": track.get("name"),
            "artist_name": track.get("artist_name"),
            "license_ccurl": track.get("license_ccurl"),
        },
    }


def cut_track_clips(
    source: Path,
    genre_dir: Path,
    track_id: str,
    *,
    genre: str,
    track: dict[str, Any],
    target_sr: int,
    mono: bool,
    clip_seconds: float,
    hop_seconds: float,
    min_seconds: float,
    max_clips: int | None,
    create_id,
    subtype: str = "PCM_24",
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Corta `source` en clips leyendo por ventanas. Devuelve (entries, rejection).
    `rejection` es None si se generó al menos un clip; si no, un dict con la razón.
    """
    import numpy as np
    import soundfile as sf

    genre_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    # --- Camino rápido en streaming con soundfile (sin resampling) ---
    try:
        with sf.SoundFile(str(source)) as snd:
            loaded_sr = snd.samplerate
            total_samples = len(snd)  # frames, sin decodificar todo
            if loaded_sr != target_sr:
                raise RuntimeError("sample_rate_mismatch")  # -> fallback librosa

            clip_samples = int(round(clip_seconds * loaded_sr))
            hop_samples = int(round(hop_seconds * loaded_sr))
            min_samples = int(round(min_seconds * loaded_sr))

            if total_samples < min_samples:
                return [], {
                    "track_id": track_id, "genre": genre, "source_audio": str(source),
                    "duration_seconds": round(total_samples / loaded_sr, 4),
                    "reason": "audio_shorter_than_min_clip",
                }

            for idx, (start, end) in enumerate(
                iter_windows(total_samples, clip_samples, hop_samples, min_samples, max_clips), start=1
            ):
                snd.seek(start)
                block = snd.read(end - start, dtype="float32", always_2d=False)  # SOLO la ventana
                if mono and getattr(block, "ndim", 1) > 1:
                    block = np.mean(block, axis=1)
                clip_path = genre_dir / f"{track_id}_clip_{idx:03d}.wav"
                sf.write(clip_path, block, loaded_sr, subtype=subtype)
                entries.append(_entry(
                    clip_id=create_id(f"{track_id}-{idx}", prefix="clip"),
                    track_id=track_id, genre=genre, clip_index=idx, source=source,
                    clip_path=clip_path, start_sample=start, end_sample=end,
                    loaded_sr=loaded_sr, track=track,
                ))
            if entries:
                return entries, None
            return [], {
                "track_id": track_id, "genre": genre, "source_audio": str(source),
                "reason": "no_clips_generated",
            }

    except (OSError, RuntimeError, ValueError):
        pass  # cae al camino librosa

    # --- Fallback: librosa leyendo POR CLIP (offset/duration), tampoco carga todo ---
    try:
        import librosa
    except ImportError as exc:
        return [], {
            "track_id": track_id, "genre": genre, "source_audio": str(source),
            "reason": f"audio_read_error: librosa requerido para resampling/MP3 ({exc})",
        }

    try:
        loaded_sr = target_sr
        total_seconds = float(librosa.get_duration(path=str(source)))
    except Exception as exc:  # noqa: BLE001
        return [], {
            "track_id": track_id, "genre": genre, "source_audio": str(source),
            "reason": f"audio_read_error: {exc}",
        }

    total_samples = int(round(total_seconds * loaded_sr))
    clip_samples = int(round(clip_seconds * loaded_sr))
    hop_samples = int(round(hop_seconds * loaded_sr))
    min_samples = int(round(min_seconds * loaded_sr))

    if total_samples < min_samples:
        return [], {
            "track_id": track_id, "genre": genre, "source_audio": str(source),
            "duration_seconds": round(total_seconds, 4),
            "reason": "audio_shorter_than_min_clip",
        }

    import soundfile as sf  # noqa: F811
    for idx, (start, end) in enumerate(
        iter_windows(total_samples, clip_samples, hop_samples, min_samples, max_clips), start=1
    ):
        offset = start / loaded_sr
        dur = (end - start) / loaded_sr
        block, _ = librosa.load(str(source), sr=loaded_sr, mono=mono, offset=offset, duration=dur)
        if not mono and getattr(block, "ndim", 1) > 1:
            block = block.T  # librosa da (canales, n); sf.write espera (n, canales)
        clip_path = genre_dir / f"{track_id}_clip_{idx:03d}.wav"
        sf.write(clip_path, block, loaded_sr, subtype=subtype)
        entries.append(_entry(
            clip_id=create_id(f"{track_id}-{idx}", prefix="clip"),
            track_id=track_id, genre=genre, clip_index=idx, source=source,
            clip_path=clip_path, start_sample=start, end_sample=end,
            loaded_sr=loaded_sr, track=track,
        ))

    if entries:
        return entries, None
    return [], {
        "track_id": track_id, "genre": genre, "source_audio": str(source),
        "reason": "no_clips_generated",
    }
