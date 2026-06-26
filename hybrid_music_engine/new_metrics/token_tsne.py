"""token_tsne.py — Adaptador de genre_tsne.py para la API/frontend.

Reutiliza las funciones de `genre_tsne` (carga de tokens EnCodec/T5, t-SNE con
PCA previo y aristas k-NN) pero, en lugar de escribir PNG/CSV, devuelve las
coordenadas 2D como JSON para que el frontend dibuje un mapa interactivo:

  • Palabras (T5)   -> encodings_v2/t5_seq/*.npz
  • Sonidos (EnCodec) -> encodings_v2/encodec/**/*_embed.npy

Cada modalidad se proyecta por separado (viven en espacios distintos) y se
colorea por genero. Es una vista EXPLORATORIA; no es una metrica de evaluacion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.new_metrics.genre_tsne import (
    GENRE_COLORS,
    Item,
    knn_edges,
    load_encodec_dir,
    load_t5_dir,
    run_tsne,
)

FALLBACK_COLORS = ["#7C3AED", "#EA580C", "#0891B2", "#CA8A04", "#DB2777", "#64748B"]


def _color_map(genres: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    extra = 0
    for genre in genres:
        if genre in GENRE_COLORS:
            colors[genre] = GENRE_COLORS[genre]
        elif genre == "unknown":
            colors[genre] = "#94A3B8"
        else:
            colors[genre] = FALLBACK_COLORS[extra % len(FALLBACK_COLORS)]
            extra += 1
    return colors


def _resolve_dirs(
    config: EngineConfig,
    t5_dir: str | None,
    encodec_dir: str | None,
) -> tuple[Path | None, Path | None, str]:
    """Resuelve los directorios de encodings. Si no se pasan, autodetecta
    `encodings_v2` y, como respaldo, `encodings_v2_test` bajo data/."""
    if t5_dir or encodec_dir:
        t5 = Path(t5_dir).expanduser() if t5_dir else None
        enc = Path(encodec_dir).expanduser() if encodec_dir else None
        return (t5 if t5 and t5.exists() else None, enc if enc and enc.exists() else None, "manual")

    for base_name in ("encodings_v2", "encodings_v2_test"):
        base = config.data_dir / base_name
        t5 = base / "t5_seq"
        enc = base / "encodec"
        if t5.exists() or enc.exists():
            return (t5 if t5.exists() else None, enc if enc.exists() else None, base_name)
    return (None, None, "")


def _subsample(items: list[Item], max_points: int, seed: int) -> list[Item]:
    if max_points <= 0 or len(items) <= max_points:
        return items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=max_points, replace=False))
    return [items[int(i)] for i in idx]


def _map_for_items(items: list[Item], *, seed: int, knn: int, draw_edges: bool) -> dict[str, Any]:
    vectors = np.vstack([item.vector for item in items]).astype(np.float64)
    coords = run_tsne(vectors, seed=seed)
    points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "genre": item.genre,
            "modality": item.modality,
            "label": item.label,
        }
        for i, item in enumerate(items)
    ]
    edges = knn_edges(vectors, k=knn) if draw_edges else []
    return {"points": points, "edges": [[int(a), int(b)] for a, b in edges], "n": len(items)}


def token_tsne_projection(
    config: EngineConfig,
    *,
    t5_dir: str | None = None,
    encodec_dir: str | None = None,
    pool_text: bool = False,
    max_points: int = 400,
    knn: int = 5,
    draw_edges: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Proyecta a 2D (t-SNE con PCA previo) los tokens EnCodec/T5 por modalidad."""
    t5_path, encodec_path, source = _resolve_dirs(config, t5_dir, encodec_dir)
    if not t5_path and not encodec_path:
        raise RuntimeError(
            "No se encontraron encodings EnCodec/T5 (carpetas encodings_v2/t5_seq o "
            "encodings_v2/encodec). Genera las encodings primero."
        )

    items: list[Item] = []
    if t5_path:
        items += load_t5_dir(t5_path, pool=pool_text)
    if encodec_path:
        items += load_encodec_dir(encodec_path)
    if not items:
        raise RuntimeError("No se cargaron tokens desde las encodings disponibles.")

    by_modality: dict[str, list[Item]] = {}
    for item in items:
        by_modality.setdefault(item.modality, []).append(item)

    maps: dict[str, Any] = {}
    for modality, group in by_modality.items():
        group = _subsample(group, max_points, seed)
        if len(group) < 3:
            maps[modality] = {"points": [], "edges": [], "n": len(group), "error": "Se necesitan >= 3 tokens."}
            continue
        maps[modality] = _map_for_items(group, seed=seed, knn=knn, draw_edges=draw_edges)

    genres = sorted({item.genre for item in items})
    modality_labels = {"word": "Palabras (T5)", "sound": "Sonidos (EnCodec)"}
    return {
        "source": source,
        "t5_dir": str(t5_path) if t5_path else None,
        "encodec_dir": str(encodec_path) if encodec_path else None,
        "modalities": list(by_modality.keys()),
        "modality_labels": modality_labels,
        "genres": genres,
        "genre_colors": _color_map(genres),
        "knn": knn,
        "maps": maps,
        "note": "t-SNE exploratorio sobre tokens EnCodec/T5; no es una metrica de evaluacion.",
    }
