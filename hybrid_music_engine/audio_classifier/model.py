
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}
DEFAULT_CLASS_LABELS = ["classical", "electronic", "reggaeton"]


def _audio_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _safe_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).any():
        return 0.0, 0.0
    values = values[np.isfinite(values)]
    return float(values.mean()), float(values.std())


def extract_audio_features(audio_path: Path, *, sample_rate: int = 22050, max_seconds: float = 45.0) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa es necesario para el clasificador de audio.") from exc

    path = Path(audio_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Audio no encontrado: {path}")
    try:
        y, sr = librosa.load(str(path), sr=sample_rate, mono=True, duration=max_seconds)
    except Exception as exc:  # librosa can raise backend-specific exceptions
        raise RuntimeError(f"No se pudo leer audio para clasificar: {path}: {exc}") from exc
    if y.size < 256 or float(np.max(np.abs(y))) < 1e-6:
        raise RuntimeError(f"Audio vacío o demasiado silencioso: {path}")

    features: list[float] = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1).astype(float).tolist())
    features.extend(np.std(mfcc, axis=1).astype(float).tolist())
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1).astype(float).tolist())
    features.extend(np.std(chroma, axis=1).astype(float).tolist())
    for matrix in [
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ]:
        features.extend(_safe_stats(matrix))
    try:
        tempo = float(np.asarray(librosa.beat.tempo(y=y, sr=sr)).ravel()[0])
    except Exception:
        tempo = 0.0
    features.append(tempo / 240.0)
    arr = np.asarray(features, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(values.shape, 1.0 / len(values), dtype=np.float64)
    return exp / total


@dataclass
class AudioCentroidClassifier:
    labels: list[str]
    centroids: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    temperature: float = 1.0

    @classmethod
    def load(cls, model_path: Path) -> "AudioCentroidClassifier":
        path = Path(model_path).expanduser()
        if path.is_dir():
            path = path / "classifier.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            labels=[str(item) for item in payload["labels"]],
            centroids=np.asarray(payload["centroids"], dtype=np.float64),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
            temperature=float(payload.get("temperature", 1.0)),
        )

    def predict_proba(self, audio_path: Path) -> list[float]:
        feature = extract_audio_features(Path(audio_path))
        normalized = (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)
        distances = np.linalg.norm(self.centroids - normalized.reshape(1, -1), axis=1)
        logits = -distances / max(self.temperature, 1e-6)
        return _softmax(logits).astype(float).tolist()

    def predict_batch(self, audio_paths: Iterable[Path]) -> np.ndarray:
        rows = [self.predict_proba(Path(path)) for path in audio_paths]
        if not rows:
            return np.empty((0, len(self.labels)), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)


def train_audio_classifier(
    config: EngineConfig,
    *,
    real_audio_root: Path,
    labels: list[str] | None = None,
    output_name: str = "audio_classifier",
    max_files_per_class: int | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    root = Path(real_audio_root).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Carpeta de audio real no encontrada: {root}")
    selected_labels = labels or sorted(p.name for p in root.iterdir() if p.is_dir())
    if not selected_labels:
        raise RuntimeError("No se encontraron carpetas de género para entrenar el clasificador.")

    features_by_label: dict[str, list[np.ndarray]] = {}
    errors: list[dict[str, str]] = []
    for label in selected_labels:
        files = _audio_files(root / label)
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        if not files:
            errors.append({"label": label, "error": "sin audios"})
            continue
        rows: list[np.ndarray] = []
        for audio in files:
            try:
                rows.append(extract_audio_features(audio))
            except RuntimeError as exc:
                errors.append({"label": label, "path": str(audio), "error": str(exc)})
        if rows:
            features_by_label[label] = rows

    if len(features_by_label) < 2:
        raise RuntimeError("Se necesitan al menos dos géneros con audio válido para entrenar el clasificador KLD.")

    labels_final = sorted(features_by_label)
    all_features = np.vstack([np.vstack(features_by_label[label]) for label in labels_final])
    feature_mean = all_features.mean(axis=0)
    feature_std = all_features.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    centroids = []
    counts = {}
    for label in labels_final:
        matrix = np.vstack(features_by_label[label])
        normalized = (matrix - feature_mean) / feature_std
        centroids.append(normalized.mean(axis=0))
        counts[label] = int(matrix.shape[0])

    model_id = create_id(output_name, prefix="classifier")
    output_dir = config.data_dir / "models" / "audio_classifier" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "classifier.json"
    payload = {
        "schema_version": "audio-centroid-classifier-v1",
        "classifier_id": model_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "real_audio_root": str(root),
        "labels": labels_final,
        "counts": counts,
        "feature_dim": int(all_features.shape[1]),
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_std": feature_std.astype(float).tolist(),
        "centroids": np.vstack(centroids).astype(float).tolist(),
        "temperature": float(temperature),
        "errors": errors[:200],
        "path": str(model_path),
    }
    model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "files"])
        writer.writeheader()
        for label in labels_final:
            writer.writerow({"label": label, "files": counts[label]})
    payload["summary_csv"] = str(summary_path)
    model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def train(real_audio_root: Path, labels: list[str] | None = None, config: EngineConfig | None = None, **kwargs) -> dict[str, Any]:
    cfg = config or EngineConfig.from_env()
    return train_audio_classifier(cfg, real_audio_root=real_audio_root, labels=labels, **kwargs)


def predict_proba(audio_path: Path, model_path: Path | None = None) -> list[float]:
    cfg = EngineConfig.from_env()
    path = model_path or _latest_classifier_path(cfg)
    return AudioCentroidClassifier.load(path).predict_proba(audio_path)


def predict_batch(audio_paths: Iterable[Path], model_path: Path | None = None) -> np.ndarray:
    cfg = EngineConfig.from_env()
    path = model_path or _latest_classifier_path(cfg)
    return AudioCentroidClassifier.load(path).predict_batch(audio_paths)


def _latest_classifier_path(config: EngineConfig) -> Path:
    candidates = sorted((config.data_dir / "models" / "audio_classifier").glob("*/classifier.json"))
    if not candidates:
        raise RuntimeError("No hay clasificador entrenado. Ejecuta classifier-train primero.")
    return candidates[-1]
