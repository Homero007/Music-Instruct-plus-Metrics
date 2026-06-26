#!/usr/bin/env python3
"""
words_tsne.py — t-SNE SOLO de palabras (tokens de texto T5) por genero.

Por instruccion del proyecto, este flujo se restringe a PALABRAS: los tokens de
texto que describen cada genero (captions / instrucciones), codificados por T5.
No incluye sonidos (EnCodec). Como todo vive en el MISMO espacio T5 (768D), las
distancias entre palabras SI son comparables y t-SNE conserva su estructura local.

Produce:
  • Un grafo por genero: palabras de ese genero + aristas k-NN (distancias locales).
  • Un grafo amplio: los 3 generos juntos, coloreados por genero.

Es EXPLORATORIO: no toca el reward model del backend (features.py/model.py/
rerank.py) ni las metricas FAD/CLAP/KLD. Solo lee artefactos y dibuja.

Fuentes de palabras
-------------------
  A) encodings_v2/t5_seq/*.npz  (lo que escribe encode_audio_text_v2):
       genre__<g>.npz   -> tokens de texto del caption del genero
       track__<key>.npz -> tokens por pista (genero via --genre-map)
     Cada fila de `hidden` (L,768) es un token-palabra.

  B) Vocabulario explicito por genero (un punto por PALABRA, mas interpretable):
       --vocab vocab.json  con  {"classical": ["piano","violin",...], ...}
     Requiere el encoder T5 para embeber cada palabra (necesita el modelo).

Uso
---
    # Desde los embeddings T5 ya calculados (recomendado)
    python words_tsne.py --t5-dir data/encodings_v2/t5_seq --out words_out --knn 6

    # Un punto por palabra, embebiendo un vocabulario por genero con T5
    python words_tsne.py --vocab vocab_generos.json --out words_out
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

# Reutilizamos las primitivas ya verificadas del modulo general.
from genre_tsne import (
    Item, run_tsne, knn_edges, plot_map, save_coords, _infer_genre, load_t5_dir,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("words_tsne")


# ── Vocabulario explicito por genero (un punto = una palabra) ─────────────────

def _split_words(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        flat: list[str] = []
        for v in value:
            flat.extend(_split_words(v))
        return flat
    return [w.strip() for w in str(value).replace("|", ",").split(",") if w.strip()]


def items_from_vocab(vocab: dict[str, object], device: str = "cpu") -> list[Item]:
    """
    Embebe cada palabra de cada genero con T5 (vector pooled por palabra).
    Un Item = una palabra. Requiere text_encoder (modelo T5).
    """
    from text_encoder import T5SequenceEncoder

    enc = T5SequenceEncoder(device=device)
    items: list[Item] = []
    for genre, words in vocab.items():
        seen: set[str] = set()
        for word in _split_words(words):
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            vec = enc.encode_pooled(word)   # (D,) un vector por palabra (mean-pool)
            items.append(Item(np.asarray(vec, dtype=np.float32), genre, "word", word))
    log.info("Vocabulario: %d palabras embebidas en %d generos",
             len(items), len(vocab))
    return items


# ── Orquestacion words-only ───────────────────────────────────────────────────

def build_word_maps(
    items: list[Item],
    out_dir: Path,
    seed: int = 42,
    draw_edges: bool = True,
    knn_k: int = 6,
) -> dict[str, Path]:
    if any(it.modality != "word" for it in items):
        raise ValueError("words_tsne solo admite modalidad 'word'.")
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}

    # 1) Mapa amplio: 3 generos juntos.
    if len(items) >= 3:
        vecs = np.vstack([it.vector for it in items])
        coords = run_tsne(vecs, seed=seed)
        edges = knn_edges(vecs, k=knn_k) if draw_edges else None
        combined = out_dir / "tsne_palabras_3generos.png"
        plot_map(coords, items, "t-SNE palabras — 3 géneros", combined,
                 edges=edges, color_by="genre")
        save_coords(coords, items, out_dir / "coords_palabras_3generos.csv")
        produced["palabras_3generos"] = combined

    # 2) Un grafo por genero (estructura local intra-genero).
    for genre in sorted({it.genre for it in items}):
        sub = [it for it in items if it.genre == genre]
        if len(sub) < 3:
            log.warning("Genero '%s': pocas palabras (%d), se omite.", genre, len(sub))
            continue
        sv = np.vstack([it.vector for it in sub])
        sc = run_tsne(sv, seed=seed)
        se = knn_edges(sv, k=knn_k) if draw_edges else None
        p = out_dir / f"tsne_palabras_{genre}.png"
        plot_map(sc, sub, f"t-SNE palabras — {genre}", p, edges=se, color_by="genre")
        save_coords(sc, sub, out_dir / f"coords_palabras_{genre}.csv")
        produced[f"palabras_{genre}"] = p

    summary = out_dir / "words_tsne_summary.json"
    with summary.open("w", encoding="utf-8") as f:
        json.dump({
            "modality": "word",
            "n_words": len(items),
            "genres": sorted({it.genre for it in items}),
            "dim": int(items[0].vector.shape[0]) if items else 0,
            "outputs": {k: str(v) for k, v in produced.items()},
            "note": "t-SNE exploratorio solo de palabras; no es metrica de evaluacion.",
        }, f, indent=2, ensure_ascii=False)
    produced["summary"] = summary
    return produced


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="t-SNE SOLO de palabras (tokens T5) por genero y combinado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--t5-dir", type=Path, default=None, help="encodings_v2/t5_seq")
    p.add_argument("--vocab", type=Path, default=None,
                   help="JSON {genero: [palabras]} para embeber un punto por palabra (usa T5).")
    p.add_argument("--genre-map", type=Path, default=None,
                   help="JSON {substring: genero} para track__/edit__ sin genero explicito.")
    p.add_argument("--out", type=Path, default=Path("words_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn", type=int, default=6)
    p.add_argument("--no-edges", action="store_true")
    p.add_argument("--pool-text", action="store_true",
                   help="Un vector por texto (media) en vez de token por token.")
    p.add_argument("--device", default="cpu")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    genre_map = json.loads(args.genre_map.read_text(encoding="utf-8")) if args.genre_map else None

    items: list[Item] = []
    if args.t5_dir:
        items += load_t5_dir(args.t5_dir, genre_map=genre_map, pool=args.pool_text)
    if args.vocab:
        vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
        items += items_from_vocab(vocab, device=args.device)

    if not items:
        raise SystemExit("Sin palabras. Indica --t5-dir y/o --vocab.")

    produced = build_word_maps(
        items, args.out, seed=args.seed,
        draw_edges=not args.no_edges, knn_k=args.knn,
    )
    print("\n=== Grafos de palabras generados ===")
    for key, path in produced.items():
        print(f"{key:>22}: {path}")


if __name__ == "__main__":
    main()
