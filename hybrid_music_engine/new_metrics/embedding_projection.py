"""embedding_projection.py — Proyecciones 2D (PCA y t-SNE) de los embeddings Token-VAE.

A partir de una corrida de *embeddings por genero* (genre_embeddings.json) se
reconstruye el espacio latente: cada PISTA del manifiesto fuente se codifica con
el mismo Token-VAE para obtener un punto latente, y ademas se incluyen los
CENTROIDES por genero (los embeddings ya guardados). Sobre esa nube se calculan
dos proyecciones a 2D:

  • PCA   — proyeccion lineal; conserva la varianza global y es interpretable.
  • t-SNE — proyeccion no lineal; resalta la estructura/los clusteres locales.

Es una vista EXPLORATORIA de como se separan los 3 generos en el espacio del
modelo. No reemplaza a FAD / KLD / CLAP / reward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.embeddings.token_vae import (
    TokenVAE,
    _resolve_token_vae_model,
    _token_histogram,
)

GENRE_COLORS = {
    "classical": "#2563EB",
    "electronic": "#DC2626",
    "reggaeton": "#059669",
}
FALLBACK_COLORS = ["#7C3AED", "#EA580C", "#0891B2", "#CA8A04", "#DB2777"]


def _color_map(genres: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    extra = 0
    for genre in genres:
        if genre in GENRE_COLORS:
            colors[genre] = GENRE_COLORS[genre]
        else:
            colors[genre] = FALLBACK_COLORS[extra % len(FALLBACK_COLORS)]
            extra += 1
    return colors


def _load_model(config: EngineConfig, model_path: Path | None):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("La proyeccion de embeddings requiere PyTorch.") from exc

    resolved = _resolve_token_vae_model(config, model_path)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    vocab = {str(key): int(value) for key, value in checkpoint["vocab"].items()}
    model = TokenVAE(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        nn_module=nn,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab, int(checkpoint["latent_dim"]), resolved


def _encode_histogram(model, vector: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        tensor = torch.tensor(vector.reshape(1, -1), dtype=torch.float32)
        mu, _ = model.encode(tensor)
    return mu.numpy().reshape(-1)


def _per_file_points(
    config: EngineConfig,
    manifest_path: Path,
    model_path: Path | None,
    max_files_per_genre: int,
) -> tuple[list[dict[str, Any]], int, str]:
    model, vocab, latent_dim, resolved = _load_model(config, model_path)
    manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    points: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for entry in manifest.get("entries", []):
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = [str(token) for token in payload.get("tokens", [])]
        if len(tokens) < 4:
            continue
        genre = str(entry.get("genre") or payload.get("genre") or "unknown").strip() or "unknown"
        if counts.get(genre, 0) >= max_files_per_genre:
            continue
        vector = _token_histogram(tokens, vocab)
        mu = _encode_histogram(model, vector)
        points.append(
            {
                "genre": genre,
                "label": path.stem,
                "kind": "track",
                "embedding": [float(value) for value in mu],
            }
        )
        counts[genre] = counts.get(genre, 0) + 1
    return points, latent_dim, str(resolved)


def _centroid_points(summary: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in summary.get("embeddings", []):
        emb_path = Path(str(item.get("path", "")))
        if not emb_path.exists():
            continue
        payload = json.loads(emb_path.read_text(encoding="utf-8"))
        embedding = payload.get("embedding")
        if not embedding:
            continue
        points.append(
            {
                "genre": str(payload.get("genre") or item.get("genre") or "unknown"),
                "label": f"centroide {payload.get('genre', '')}".strip(),
                "kind": "centroid",
                "embedding": [float(value) for value in embedding],
            }
        )
    return points


def _project(points: list[dict[str, Any]], seed: int) -> tuple[dict[str, list], list[float]]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    result: dict[str, list] = {"pca": [], "tsne": []}
    explained: list[float] = []
    vectors = np.array([p["embedding"] for p in points], dtype=np.float64)
    n = vectors.shape[0]
    if n < 2:
        return result, explained

    scaled = StandardScaler().fit_transform(vectors) if vectors.shape[1] > 1 else vectors
    comps = min(2, scaled.shape[1])
    pca = PCA(n_components=comps, random_state=seed).fit(scaled)
    pca_coords = pca.transform(scaled)
    explained = [float(value) for value in pca.explained_variance_ratio_]
    for index, point in enumerate(points):
        result["pca"].append(
            {
                "x": float(pca_coords[index, 0]),
                "y": float(pca_coords[index, 1]) if comps > 1 else 0.0,
                "genre": point["genre"],
                "label": point["label"],
                "kind": point["kind"],
            }
        )

    if n >= 3:
        from hybrid_music_engine.new_metrics.genre_tsne import run_tsne

        tsne_coords = run_tsne(vectors, seed=seed)
        for index, point in enumerate(points):
            result["tsne"].append(
                {
                    "x": float(tsne_coords[index, 0]),
                    "y": float(tsne_coords[index, 1]),
                    "genre": point["genre"],
                    "label": point["label"],
                    "kind": point["kind"],
                }
            )
    return result, explained


def embedding_projection_from_run(
    config: EngineConfig,
    run_path: Path,
    *,
    seed: int = 42,
    max_files_per_genre: int = 80,
) -> dict[str, Any]:
    """Proyecta a 2D (PCA y t-SNE) los embeddings de una corrida por genero."""
    summary = json.loads(Path(run_path).expanduser().read_text(encoding="utf-8"))
    manifest_path = summary.get("source_manifest_path")
    if not manifest_path or not Path(manifest_path).expanduser().exists():
        raise RuntimeError(
            "No se encontro el manifiesto de tokens fuente de esta corrida. "
            "Vuelve a generar los embeddings por genero."
        )
    model_path = summary.get("model_path")
    model_path = Path(model_path) if model_path else None

    points, latent_dim, model_used = _per_file_points(
        config, Path(manifest_path), model_path, max_files_per_genre
    )
    points.extend(_centroid_points(summary))
    if len(points) < 2:
        raise RuntimeError("No hay suficientes embeddings para proyectar (se necesitan >= 2).")

    projections, explained = _project(points, seed)
    genres = sorted({point["genre"] for point in points})
    return {
        "run_id": summary.get("run_id"),
        "model_path": model_used,
        "latent_dim": latent_dim,
        "n_points": len(points),
        "n_tracks": sum(1 for point in points if point["kind"] == "track"),
        "n_centroids": sum(1 for point in points if point["kind"] == "centroid"),
        "genres": genres,
        "genre_colors": _color_map(genres),
        "pca_explained_variance": explained,
        "has_tsne": bool(projections["tsne"]),
        "pca": projections["pca"],
        "tsne": projections["tsne"],
    }
