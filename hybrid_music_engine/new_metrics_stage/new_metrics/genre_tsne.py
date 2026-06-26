#!/usr/bin/env python3
"""
genre_tsne.py — Mapas t-SNE de tokens por genero (sonidos y palabras).

Objetivo
--------
Visualizar, conservando la ESTRUCTURA LOCAL, como se distribuyen los tokens de
cada genero y como se separan/solapan los 3 generos entre si. Produce:

  1. Un grafo por genero con sus tokens (scatter t-SNE + aristas k-NN opcionales
     que dibujan las distancias locales = "grafo con la distancia entre tokens").
  2. Un grafo amplio con los 3 generos juntos, coloreados por genero.

Esto es EXPLORATORIO. NO es una metrica de evaluacion: no toca ni reemplaza a
FAD / CLAP / KLD / reward_model. Solo lee artefactos y dibuja.

Fuentes de tokens (formatos que ya escribe el proyecto)
-------------------------------------------------------
  • Palabras (texto):  encodings_v2/t5_seq/<key>.npz   con clave `hidden` (L,768)
        genre__<g>.npz   -> tokens de texto del caption del genero
        track__<key>.npz -> tokens de texto por pista (genero via --genre-map)
  • Sonidos (audio):   encodings_v2/encodec/<rel>/<stem>_embed.npy  (n_frames,dim)
        el genero se infiere del primer componente de <rel> (p.ej. .../electronic/..)

Nota sobre "sonidos + palabras en un mismo grafo"
-------------------------------------------------
T5 (768) y EnCodec (dim distinto) viven en ESPACIOS DISTINTOS: la distancia entre
una palabra-token y un sonido-token NO es metricamente comparable. Por eso:
  • Por defecto se dibuja un mapa POR MODALIDAD (palabras y sonidos por separado).
  • Para un mapa CONJUNTO valido de sonidos+palabras usa un espacio COMPARTIDO
    (CLAP), pasando --joint-shared con .npz/.npy que ya esten en ese espacio comun.
    CLAP coloca audio y texto en el mismo espacio, asi que ahi la distancia si
    significa algo.

Uso
---
    # Palabras (T5) por genero + mapa de los 3 generos
    python genre_tsne.py --t5-dir data/encodings_v2/t5_seq --out tsne_out

    # Sonidos (EnCodec) por genero
    python genre_tsne.py --encodec-dir data/encodings_v2/encodec --out tsne_out

    # Ambos: dos capas (palabras y sonidos), cada una en su propio mapa
    python genre_tsne.py --t5-dir .../t5_seq --encodec-dir .../encodec --out tsne_out

Como modulo
-----------
    from genre_tsne import Item, run_tsne, plot_map
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("genre_tsne")

KNOWN_GENRES = ("classical", "electronic", "reggaeton")
GENRE_COLORS = {
    "classical": "#2563EB",
    "electronic": "#DC2626",
    "reggaeton": "#059669",
}
FALLBACK_COLORS = ["#7C3AED", "#EA580C", "#0891B2", "#CA8A04", "#DB2777"]


@dataclass
class Item:
    """Un token proyectable: su vector, su genero y su modalidad."""
    vector: np.ndarray   # (D,)
    genre: str
    modality: str        # "word" | "sound"
    label: str = ""      # texto/origen para trazabilidad


# ── Carga desde artefactos del proyecto ───────────────────────────────────────

def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_").lower()


def _infer_genre(text: str, genre_map: dict[str, str] | None = None) -> str:
    low = text.lower()
    for g in KNOWN_GENRES:
        if g in low:
            return g
    if genre_map:
        for key, g in genre_map.items():
            if key.lower() in low:
                return g
    return "unknown"


def load_t5_dir(
    t5_dir: Path,
    genre_map: dict[str, str] | None = None,
    pool: bool = False,
    max_tokens_per_file: int = 64,
) -> list[Item]:
    """
    Lee encodings_v2/t5_seq/*.npz. Cada archivo aporta sus L tokens de texto
    (filas de `hidden`). Si pool=True, aporta un solo vector (media) por archivo.
    """
    items: list[Item] = []
    files = sorted(t5_dir.glob("*.npz"))
    if not files:
        log.warning("Sin .npz en %s", t5_dir)
    for f in files:
        data = np.load(f, allow_pickle=False)
        hidden = data["hidden"].astype(np.float32)        # (L, 768)
        text = str(data["text"]) if "text" in data else f.stem
        stem = f.stem
        if stem.startswith("genre__"):
            genre = stem[len("genre__"):]
        elif stem.startswith("track__") or stem.startswith("edit__"):
            genre = _infer_genre(text, genre_map)
        else:
            genre = _infer_genre(stem, genre_map)

        if pool:
            items.append(Item(hidden.mean(axis=0), genre, "word", text[:60]))
        else:
            rows = hidden
            if rows.shape[0] > max_tokens_per_file:
                idx = np.linspace(0, rows.shape[0] - 1, max_tokens_per_file).astype(int)
                rows = rows[idx]
            for r in rows:
                items.append(Item(r, genre, "word", text[:40]))
    log.info("T5: %d tokens-palabra desde %d archivos", len(items), len(files))
    return items


def load_encodec_dir(
    encodec_dir: Path,
    max_frames_per_file: int = 48,
    genre_map: dict[str, str] | None = None,
) -> list[Item]:
    """
    Lee encodings_v2/encodec/**/<stem>_embed.npy (n_frames, dim). Submuestrea
    frames por archivo. El genero se infiere de la ruta relativa.
    """
    items: list[Item] = []
    files = sorted(encodec_dir.rglob("*_embed.npy"))
    if not files:
        log.warning("Sin *_embed.npy en %s", encodec_dir)
    for f in files:
        emb = np.load(f).astype(np.float32)               # (n_frames, dim)
        if emb.ndim != 2:
            log.warning("Saltando %s: shape %s", f.name, emb.shape)
            continue
        rel = f.relative_to(encodec_dir)
        genre = _infer_genre(str(rel), genre_map)
        rows = emb
        if rows.shape[0] > max_frames_per_file:
            idx = np.linspace(0, rows.shape[0] - 1, max_frames_per_file).astype(int)
            rows = rows[idx]
        for r in rows:
            items.append(Item(r, genre, "sound", f.stem))
    log.info("EnCodec: %d tokens-sonido desde %d archivos", len(items), len(files))
    return items


def load_shared_npz(paths: Sequence[Path], modality: str, genre_map=None) -> list[Item]:
    """Carga vectores de un espacio COMPARTIDO (p.ej. CLAP) desde .npy/.npz."""
    items: list[Item] = []
    for p in paths:
        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            arr = data[data.files[0]].astype(np.float32)
        else:
            arr = np.load(p).astype(np.float32)
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)
        genre = _infer_genre(p.stem, genre_map)
        for r in arr:
            items.append(Item(r, genre, modality, p.stem))
    return items


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def _auto_perplexity(n: int) -> float:
    # perplexity debe ser < n; regla practica 5..30 acotada por (n-1)/3.
    return float(max(5.0, min(30.0, (n - 1) / 3.0)))


def run_tsne(
    vectors: np.ndarray,
    seed: int = 42,
    perplexity: float | None = None,
    pca_dims: int = 50,
) -> np.ndarray:
    """
    Proyecta (N, D) -> (N, 2) con t-SNE, conservando estructura local.
    Estandariza, reduce con PCA a `pca_dims` si D es grande (practica estandar),
    y luego aplica t-SNE.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = vectors.shape[0]
    if n < 3:
        raise ValueError(f"t-SNE necesita >=3 puntos, hay {n}.")

    X = StandardScaler().fit_transform(vectors.astype(np.float64))
    if X.shape[1] > pca_dims and X.shape[1] > 2:
        comps = min(pca_dims, n - 1, X.shape[1])
        X = PCA(n_components=comps, random_state=seed).fit_transform(X)

    perp = perplexity if perplexity is not None else _auto_perplexity(n)
    perp = min(perp, max(2.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2, perplexity=perp, init="pca",
        learning_rate="auto", random_state=seed, max_iter=1000,
    )
    return tsne.fit_transform(X)


def knn_edges(vectors: np.ndarray, k: int = 5) -> list[tuple[int, int]]:
    """Aristas k-NN en el espacio ORIGINAL (las distancias que t-SNE preserva)."""
    from sklearn.neighbors import NearestNeighbors

    n = vectors.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(vectors)
    _, idx = nn.kneighbors(vectors)
    edges = set()
    for i in range(n):
        for j in idx[i, 1:]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


# ── Graficado ─────────────────────────────────────────────────────────────────

def _color_for(genre: str, palette: dict[str, str]) -> str:
    if genre in GENRE_COLORS:
        return GENRE_COLORS[genre]
    if genre not in palette:
        palette[genre] = FALLBACK_COLORS[len(palette) % len(FALLBACK_COLORS)]
    return palette[genre]


def plot_map(
    coords: np.ndarray,
    items: list[Item],
    title: str,
    path: Path,
    edges: list[tuple[int, int]] | None = None,
    color_by: str = "genre",
) -> None:
    import matplotlib.pyplot as plt

    palette: dict[str, str] = {}
    markers = {"word": "o", "sound": "^"}
    fig, ax = plt.subplots(figsize=(9, 7))

    if edges:
        for i, j in edges:
            ax.plot(
                [coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color="#9CA3AF", linewidth=0.3, alpha=0.4, zorder=1,
            )

    seen_labels = set()
    for k, it in enumerate(items):
        if color_by == "genre":
            color = _color_for(it.genre, palette)
            legend_key = it.genre
        else:  # color por modalidad
            color = "#2563EB" if it.modality == "word" else "#DC2626"
            legend_key = it.modality
        marker = markers.get(it.modality, "o")
        label = legend_key if legend_key not in seen_labels else None
        seen_labels.add(legend_key)
        ax.scatter(
            coords[k, 0], coords[k, 1], c=color, marker=marker, s=28,
            alpha=0.8, edgecolors="white", linewidths=0.3, zorder=2, label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    log.info("Grafo -> %s", path)


def save_coords(coords: np.ndarray, items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "genre", "modality", "label"])
        for (x, y), it in zip(coords, items):
            w.writerow([f"{x:.5f}", f"{y:.5f}", it.genre, it.modality, it.label])


# ── Orquestacion de alto nivel ────────────────────────────────────────────────

def build_maps(
    items: list[Item],
    out_dir: Path,
    seed: int = 42,
    draw_edges: bool = True,
    knn_k: int = 5,
) -> dict[str, Path]:
    """
    Genera, por modalidad disponible:
      - un mapa combinado de los 3 generos (color = genero),
      - un mapa por genero (con aristas k-NN).
    Las modalidades NO se mezclan en un mismo t-SNE salvo que compartan espacio
    (mismo D). Si comparten D, se ofrece ademas un mapa conjunto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}

    by_mod: dict[str, list[Item]] = {}
    for it in items:
        by_mod.setdefault(it.modality, []).append(it)

    dims = {it.vector.shape[0] for it in items}
    shared_space = len(dims) == 1 and len(by_mod) > 1

    for modality, group in by_mod.items():
        if len(group) < 3:
            log.warning("Modalidad '%s': muy pocos tokens (%d), se omite.", modality, len(group))
            continue
        vecs = np.vstack([it.vector for it in group])
        coords = run_tsne(vecs, seed=seed)
        edges = knn_edges(vecs, k=knn_k) if draw_edges else None

        combined = out_dir / f"tsne_{modality}_3generos.png"
        plot_map(coords, group, f"t-SNE {modality} — 3 géneros", combined,
                 edges=edges, color_by="genre")
        save_coords(coords, group, out_dir / f"coords_{modality}_3generos.csv")
        produced[f"{modality}_3generos"] = combined

        # Un grafo por genero (estructura local intra-genero).
        for genre in sorted({it.genre for it in group}):
            sub = [it for it in group if it.genre == genre]
            if len(sub) < 3:
                continue
            sv = np.vstack([it.vector for it in sub])
            sc = run_tsne(sv, seed=seed)
            se = knn_edges(sv, k=knn_k) if draw_edges else None
            p = out_dir / f"tsne_{modality}_{genre}.png"
            plot_map(sc, sub, f"t-SNE {modality} — {genre}", p,
                     edges=se, color_by="modality")
            produced[f"{modality}_{genre}"] = p

    # Mapa CONJUNTO sonidos+palabras solo si comparten espacio (CLAP).
    if shared_space:
        vecs = np.vstack([it.vector for it in items])
        coords = run_tsne(vecs, seed=seed)
        edges = knn_edges(vecs, k=knn_k) if draw_edges else None
        p = out_dir / "tsne_conjunto_sonidos_palabras.png"
        plot_map(coords, items, "t-SNE conjunto — sonidos + palabras (espacio compartido)",
                 p, edges=edges, color_by="genre")
        save_coords(coords, items, out_dir / "coords_conjunto.csv")
        produced["conjunto"] = p
    elif len(by_mod) > 1:
        log.info(
            "Sonidos y palabras estan en espacios distintos (dims=%s): se grafican "
            "por separado. Para un mapa conjunto valido usa --joint-shared (CLAP).",
            sorted(dims),
        )

    summary = out_dir / "tsne_summary.json"
    with summary.open("w", encoding="utf-8") as f:
        json.dump({
            "n_items": len(items),
            "modalities": {m: len(g) for m, g in by_mod.items()},
            "genres": sorted({it.genre for it in items}),
            "dims": sorted(dims),
            "shared_space": shared_space,
            "outputs": {k: str(v) for k, v in produced.items()},
            "note": "t-SNE exploratorio; no es una metrica de evaluacion.",
        }, f, indent=2, ensure_ascii=False)
    produced["summary"] = summary
    return produced


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mapas t-SNE de tokens (sonidos/palabras) por genero y combinados.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--t5-dir", type=Path, default=None, help="encodings_v2/t5_seq")
    p.add_argument("--encodec-dir", type=Path, default=None, help="encodings_v2/encodec")
    p.add_argument("--joint-shared", type=Path, nargs="+", default=None,
                   help="Vectores en espacio compartido (CLAP) .npy/.npz para mapa conjunto.")
    p.add_argument("--genre-map", type=Path, default=None,
                   help="JSON {substring_clave: genero} para track__/edit__ sin genero explicito.")
    p.add_argument("--out", type=Path, default=Path("tsne_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn", type=int, default=5, help="Vecinos para aristas del grafo.")
    p.add_argument("--no-edges", action="store_true", help="No dibujar aristas k-NN.")
    p.add_argument("--pool-text", action="store_true",
                   help="Un vector por texto (media) en vez de token por token.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    genre_map = None
    if args.genre_map:
        genre_map = json.loads(args.genre_map.read_text(encoding="utf-8"))

    items: list[Item] = []
    if args.t5_dir:
        items += load_t5_dir(args.t5_dir, genre_map=genre_map, pool=args.pool_text)
    if args.encodec_dir:
        items += load_encodec_dir(args.encodec_dir, genre_map=genre_map)
    if args.joint_shared:
        items += load_shared_npz(args.joint_shared, modality="word", genre_map=genre_map)

    if not items:
        raise SystemExit("No se cargaron tokens. Indica --t5-dir y/o --encodec-dir.")

    produced = build_maps(
        items, args.out, seed=args.seed,
        draw_edges=not args.no_edges, knn_k=args.knn,
    )
    print("\n=== Grafos generados ===")
    for key, path in produced.items():
        print(f"{key:>22}: {path}")


if __name__ == "__main__":
    main()
