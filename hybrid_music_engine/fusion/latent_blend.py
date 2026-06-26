from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id


def blend_embeddings(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha * np.asarray(a, dtype=np.float32)) + ((1.0 - alpha) * np.asarray(b, dtype=np.float32))


def blend_embedding_files(
    config: EngineConfig,
    *,
    embedding_a_path: Path,
    embedding_b_path: Path,
    alpha: float,
    output_name: str = "latent_blend",
) -> dict[str, Any]:
    payload_a = json.loads(Path(embedding_a_path).expanduser().read_text(encoding="utf-8"))
    payload_b = json.loads(Path(embedding_b_path).expanduser().read_text(encoding="utf-8"))
    embedding_a = np.asarray(payload_a.get("embedding", []), dtype=np.float32)
    embedding_b = np.asarray(payload_b.get("embedding", []), dtype=np.float32)
    if embedding_a.size == 0 or embedding_b.size == 0:
        raise RuntimeError("Los embeddings deben contener vectores no vacíos.")
    if embedding_a.shape != embedding_b.shape:
        raise RuntimeError("Los embeddings deben tener la misma dimensión.")
    blended = blend_embeddings(embedding_a, embedding_b, alpha)
    blend_id = create_id(output_name, prefix="blend")
    output_dir = config.data_dir / "embeddings" / "blends"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{blend_id}.json"
    payload = {
        "schema_version": "latent-blend-v1",
        "blend_id": blend_id,
        "output_name": output_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alpha": float(np.clip(alpha, 0.0, 1.0)),
        "embedding_a_path": str(Path(embedding_a_path).expanduser().resolve()),
        "embedding_b_path": str(Path(embedding_b_path).expanduser().resolve()),
        "latent_dim": int(blended.size),
        "embedding": [round(float(value), 8) for value in blended.tolist()],
        "path": str(output_path),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def blend_weighted_embedding_files(
    config: EngineConfig,
    *,
    embeddings: list[dict[str, Any]],
    output_name: str = "genre_fusion",
) -> dict[str, Any]:
    if len(embeddings) < 2:
        raise RuntimeError("La fusión requiere al menos dos embeddings.")

    loaded: list[dict[str, Any]] = []
    total_weight = 0.0
    expected_shape: tuple[int, ...] | None = None
    for item in embeddings:
        path = Path(str(item.get("path", ""))).expanduser()
        weight = float(item.get("weight", 0.0))
        if weight <= 0:
            raise RuntimeError("Todos los pesos de fusión deben ser mayores a cero.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        vector = np.asarray(payload.get("embedding", []), dtype=np.float32)
        if vector.size == 0:
            raise RuntimeError(f"Embedding vacío: {path}")
        if expected_shape is None:
            expected_shape = vector.shape
        elif vector.shape != expected_shape:
            raise RuntimeError("Todos los embeddings deben tener la misma dimensión.")
        total_weight += weight
        loaded.append(
            {
                "path": str(path.resolve()),
                "weight": weight,
                "label": item.get("label") or payload.get("genre") or payload.get("embedding_id") or path.stem,
                "genre": payload.get("genre"),
                "embedding_id": payload.get("embedding_id"),
                "vector": vector,
            }
        )

    if total_weight <= 0:
        raise RuntimeError("La suma de pesos debe ser mayor a cero.")

    blended = np.zeros_like(loaded[0]["vector"], dtype=np.float32)
    sources = []
    for item in loaded:
        normalized_weight = float(item["weight"] / total_weight)
        blended += normalized_weight * item["vector"]
        sources.append(
            {
                "path": item["path"],
                "label": item["label"],
                "genre": item["genre"],
                "embedding_id": item["embedding_id"],
                "weight": round(float(item["weight"]), 8),
                "normalized_weight": round(normalized_weight, 8),
            }
        )

    blend_id = create_id(output_name, prefix="blend")
    output_dir = config.data_dir / "embeddings" / "blends"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{blend_id}.json"
    payload = {
        "schema_version": "weighted-latent-blend-v1",
        "blend_id": blend_id,
        "output_name": output_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": sources,
        "latent_dim": int(blended.size),
        "embedding": [round(float(value), 8) for value in blended.tolist()],
        "path": str(output_path),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
