#!/usr/bin/env python3
"""make_synthetic_results.py — Genera CSVs sintéticos con el esquema del PDF.

Simula 4 modelos evaluados sobre 100 prompts de MusicCaps (10 géneros × 10).
Produce:
  results/set_level_metrics.csv   — una fila por modelo
  results/clip_level_metrics.csv  — una fila por clip (400 total)

Los valores son reproducibles (seed fijo) y coinciden con los rangos del PDF:
  MusicGen-medium      FAD=1.42  CLAP=0.318±0.05  KLD=0.18  KAD=0.021
  MusicGen-small       FAD=1.95  CLAP=0.280±0.06  KLD=0.27  KAD=0.034
  AudioLDM2            FAD=2.71  CLAP=0.234±0.06  KLD=0.41  KAD=0.052
  Stable-Audio-Open    FAD=3.15  CLAP=0.221±0.05  KLD=0.52  KAD=0.061
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

MODELS = [
    {"name": "MusicGen-medium",  "fad": 1.42, "clap_mean": 0.318, "clap_std": 0.05, "kld": 0.18, "kad": 0.021},
    {"name": "MusicGen-small",   "fad": 1.95, "clap_mean": 0.280, "clap_std": 0.06, "kld": 0.27, "kad": 0.034},
    {"name": "AudioLDM2",        "fad": 2.71, "clap_mean": 0.234, "clap_std": 0.06, "kld": 0.41, "kad": 0.052},
    {"name": "Stable-Audio-Open", "fad": 3.15, "clap_mean": 0.221, "clap_std": 0.05, "kld": 0.52, "kad": 0.061},
]

GENRES = [
    "classical", "electronic", "reggaeton", "jazz", "pop",
    "rock", "ambient", "hip-hop", "latin", "world",
]

PROMPTS_PER_GENRE = 10


def make_synthetic(out_dir: Path, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    set_rows: list[dict] = []
    clip_rows: list[dict] = []

    for model in MODELS:
        all_clap: list[float] = []

        for genre in GENRES:
            for prompt_idx in range(1, PROMPTS_PER_GENRE + 1):
                # CLAP varía ligeramente por género para que Kruskal-Wallis sea interesante
                genre_offset = (GENRES.index(genre) - len(GENRES) / 2) * 0.008
                clap = rng.gauss(model["clap_mean"] + genre_offset, model["clap_std"])
                clap = max(0.0, min(1.0, clap))
                kld_clip = max(0.0, rng.gauss(model["kld"], model["kld"] * 0.15))
                tempo = rng.uniform(80, 160)
                clip_id = f"{model['name'].lower().replace(' ', '-')}_{genre}_{prompt_idx:02d}"
                clip_rows.append({
                    "model": model["name"],
                    "clip_id": clip_id,
                    "genre": genre,
                    "tempo_bpm": f"{tempo:.3f}",
                    "clap_score": f"{clap:.6f}",
                    "passt_kld": f"{kld_clip:.6f}",
                })
                all_clap.append(clap)

        mean_clap = sum(all_clap) / len(all_clap)
        variance = sum((x - mean_clap) ** 2 for x in all_clap) / len(all_clap)
        std_clap = variance ** 0.5
        pct_above = sum(1 for c in all_clap if c > 0.25) / len(all_clap) * 100
        set_rows.append({
            "model": model["name"],
            "fad_vggish": f"{model['fad']:.4f}",
            "fad_pann": f"{model['fad'] * 0.87:.4f}",
            "mean_clap": f"{mean_clap:.6f}",
            "std_clap": f"{std_clap:.6f}",
            "pct_clap_above_025": f"{pct_above:.2f}",
            "kld": f"{model['kld']:.6f}",
            "kad": f"{model['kad']:.6f}",
        })

    set_path = out_dir / "set_level_metrics.csv"
    with set_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(set_rows[0].keys()))
        writer.writeheader()
        writer.writerows(set_rows)

    clip_path = out_dir / "clip_level_metrics.csv"
    with clip_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(clip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clip_rows)

    print(f"Generados: {set_path}  ({len(set_rows)} modelos)")
    print(f"Generados: {clip_path}  ({len(clip_rows)} clips)")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    make_synthetic(root / "results")
