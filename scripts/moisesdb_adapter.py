#!/usr/bin/env python3
"""moisesdb_adapter.py — Carga de stems REALES de MoisesDB.

MoisesDB (240 canciones con stems reales multipista) elimina el sesgo de los
stems *estimados* con Demucs en el banco de edición: los objetivos de operaciones
de stems (quitar batería, aislar bajo, ...) se construyen con audio de capas
reales en vez de separaciones aproximadas.

Lee la estructura cruda del dataset (sin depender del paquete `moisesdb`):

  <root>/<track_id>/
      data.json                 (metadata: genre, stems, ...)
      drums/*.wav   bass/*.wav   vocals/*.wav
      guitar/*.wav  piano/*.wav  other/*.wav  percussion/*.wav  ...

Cada subcarpeta puede contener varios .wav (sub-stems) que se SUMAN para formar
la capa. Se mapean las categorías finas de MoisesDB a la taxonomía de
operaciones {drums, bass, vocals, other} y se normaliza a formato canónico
(32 kHz mono, 10 s).

Estructura real (MoisesDB v0.1): <root>/<provider>/<track_id>/ con un
data.json por pista y subcarpetas por categoría que pueden anidar track-types.
El descubrimiento de pistas se hace por data.json (robusto a la anidación por
provider) y la suma de cada capa es recursiva.

Descarga (gated, requiere aceptar términos):
  https://music.ai/research/   ·   https://github.com/moises-ai/moises-db
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

TARGET_SR = 32000
TARGET_LEN = TARGET_SR * 10

# Categorías finas de MoisesDB -> taxonomía de operaciones del banco.
MOISESDB_STEM_MAP: dict[str, list[str]] = {
    "drums": ["drums", "percussion"],
    "bass": ["bass"],
    "vocals": ["vocals"],
    "other": ["guitar", "piano", "other", "other_keys", "other_plucked",
              "bowed_strings", "wind"],
}
KNOWN_CATEGORIES = sorted({c for cats in MOISESDB_STEM_MAP.values() for c in cats})
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")


def _fit(y: np.ndarray, n: int = TARGET_LEN) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.shape[-1] < n:
        return np.pad(y, (0, n - y.shape[-1]))
    return y[..., :n]


def _load_mono_native(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Lee a mono en su SR nativo. `max_seconds` limita a la ventana inicial."""
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        frames = int(max_seconds * sr) if max_seconds else -1
        y = f.read(frames=frames if frames > 0 else -1, dtype="float64", always_2d=False)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def list_tracks(root: Path | str) -> list[Path]:
    """Carpetas de pista, robusto a la anidación por ``provider``/``moisesdb_v0.1``.

    Cada pista tiene un ``data.json``; se descubren por ese marcador (a cualquier
    profundidad). Fallback: directorios con al menos una categoría de stem.
    """
    root = Path(root)
    by_marker = sorted({p.parent for p in root.rglob("data.json")})
    if by_marker:
        return by_marker
    tracks = {d for d in root.rglob("*")
              if d.is_dir() and any((d / cat).is_dir() for cat in KNOWN_CATEGORIES)}
    return sorted(tracks)


def _sum_category_native(track_dir: Path, categories: list[str],
                         max_seconds: float | None) -> tuple[np.ndarray | None, int | None]:
    """Suma (recursivamente) los audios de las categorías, en SR nativo común."""
    acc: np.ndarray | None = None
    out_sr: int | None = None
    for cat in categories:
        cat_dir = track_dir / cat
        if not cat_dir.is_dir():
            continue
        for wav in sorted(cat_dir.rglob("*")):
            if wav.suffix.lower() not in AUDIO_EXTS:
                continue
            y, sr = _load_mono_native(wav, max_seconds)
            if out_sr is None:
                out_sr = sr
            elif sr != out_sr:
                import librosa
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=out_sr).astype(np.float64)
            if acc is None:
                acc = y
            else:
                n = max(acc.shape[-1], y.shape[-1])
                acc = np.pad(acc, (0, n - acc.shape[-1])) + np.pad(y, (0, n - y.shape[-1]))
    return acc, out_sr


def _best_offset(mixture: np.ndarray, sr: int, window_s: int, segment: str) -> int:
    """Offset (muestras) de la ventana de `window_s` a extraer.

    segment: ``start`` (inicio), ``middle`` (centro) o ``energy`` (ventana de
    mayor RMS, evita intros dispersos y recupera pistas).
    """
    win = int(window_s * sr)
    if mixture is None or mixture.shape[-1] <= win:
        return 0
    if segment == "start":
        return 0
    if segment == "middle":
        return (mixture.shape[-1] - win) // 2
    hop = max(1, win // 2)
    offsets = range(0, mixture.shape[-1] - win + 1, hop)
    return max(offsets, key=lambda o: float(np.mean(mixture[o:o + win] ** 2)))


def _to_canonical(y_native: np.ndarray | None, sr: int, off: int, win: int) -> np.ndarray:
    seg = y_native[off:off + win] if y_native is not None else np.zeros(win)
    if sr and sr != TARGET_SR:
        import librosa
        seg = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR).astype(np.float64)
    return _fit(seg)


def load_track_stems(track_dir: Path, segment: str = "energy", window_s: int = 10) -> dict[str, Any]:
    """Devuelve {'stems': {drums,bass,vocals,other}, 'mixture': array} canónicos.

    `segment` elige la ventana de `window_s`: ``start`` (lectura rápida del inicio),
    ``middle`` o ``energy`` (ventana de mayor energía; lee la canción completa).
    """
    max_s = (window_s + 1.0) if segment == "start" else None
    raw = {name: _sum_category_native(track_dir, cats, max_s) for name, cats in MOISESDB_STEM_MAP.items()}

    srs = [sr for (_, sr) in raw.values() if sr]
    if not srs:
        zeros = {k: np.zeros(TARGET_LEN) for k in MOISESDB_STEM_MAP}
        return {"stems": zeros, "mixture": np.zeros(TARGET_LEN)}
    sr = srs[0]

    # Alinear cada capa al SR nativo común.
    stems_native: dict[str, np.ndarray | None] = {}
    for name, (y, s) in raw.items():
        if y is not None and s != sr:
            import librosa
            y = librosa.resample(y.astype(np.float32), orig_sr=s, target_sr=sr).astype(np.float64)
        stems_native[name] = y
    present = [v for v in stems_native.values() if v is not None]
    n = max((v.shape[-1] for v in present), default=int(window_s * sr))
    stems_native = {k: (np.pad(v, (0, n - v.shape[-1])) if v is not None else np.zeros(n))
                    for k, v in stems_native.items()}

    mixture_native = sum(stems_native.values())
    win = int(window_s * sr)
    off = _best_offset(mixture_native, sr, window_s, segment)
    stems = {k: _to_canonical(v, sr, off, win) for k, v in stems_native.items()}
    mixture = _fit(sum(stems.values()))
    return {"stems": stems, "mixture": mixture}


def track_genre(track_dir: Path) -> str:
    data = track_dir / "data.json"
    if data.exists():
        try:
            return str(json.loads(data.read_text(encoding="utf-8")).get("genre", "unknown"))
        except (ValueError, OSError):
            return "unknown"
    return "unknown"
