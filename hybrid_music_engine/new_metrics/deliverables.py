#!/usr/bin/env python3
"""
deliverables.py — Entregables tabulares de la Fase 4.

Escribe los dos CSV del protocolo (sin depender de pandas):

  results/metrics_all.csv
      columnas: model, clip_id, genre, tempo_bpm, clap_score, passt_kld
      (una fila por clip; passt_kld es a nivel de conjunto y se repite por fila
       del mismo modelo, ya que KLD es distribucional —no por clip—)

  results/set_level_metrics.csv
      columnas: model, fad_vggish, fad_pann, mean_clap, std_clap,
                pct_clap_above_025, kld, kad

Las funciones aceptan estructuras simples para poder alimentarse desde cualquiera
de los dos pipelines (new_metrics o evaluation).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CLIP_COLUMNS = ["model", "clip_id", "genre", "tempo_bpm", "clap_score", "passt_kld"]
SET_COLUMNS = [
    "model", "fad_vggish", "fad_pann", "mean_clap", "std_clap",
    "pct_clap_above_025", "kld", "kad",
]


def _fmt(value, ndigits: int = 6):
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return ""
    return round(f, ndigits)


def write_metrics_all(rows: Iterable[Mapping[str, object]], out_path: Path) -> Path:
    """
    rows: iterable de dicts con claves de CLIP_COLUMNS (las faltantes quedan
    vacías). Devuelve la ruta escrita.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIP_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row.get("model", ""),
                "clip_id": row.get("clip_id", ""),
                "genre": row.get("genre", ""),
                "tempo_bpm": _fmt(row.get("tempo_bpm"), 3),
                "clap_score": _fmt(row.get("clap_score"), 6),
                "passt_kld": _fmt(row.get("passt_kld"), 6),
            })
    return out_path


def write_set_level(rows: Iterable[Mapping[str, object]], out_path: Path) -> Path:
    """rows: iterable de dicts con claves de SET_COLUMNS."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row.get("model", ""),
                "fad_vggish": _fmt(row.get("fad_vggish")),
                "fad_pann": _fmt(row.get("fad_pann")),
                "mean_clap": _fmt(row.get("mean_clap")),
                "std_clap": _fmt(row.get("std_clap")),
                "pct_clap_above_025": _fmt(row.get("pct_clap_above_025"), 2),
                "kld": _fmt(row.get("kld")),
                "kad": _fmt(row.get("kad")),
            })
    return out_path


def build_clip_rows(
    model: str,
    *,
    clip_ids: Sequence[str],
    genres: Mapping[str, str] | None = None,
    tempos: Mapping[str, float] | None = None,
    clap_scores: Mapping[str, float] | None = None,
    passt_kld: float | None = None,
) -> list[dict]:
    """
    Construye filas por clip para un modelo. `passt_kld` (escalar de conjunto) se
    repite en cada fila del modelo. Las claves de los mappings son clip_id.
    """
    genres = genres or {}
    tempos = tempos or {}
    clap_scores = clap_scores or {}
    rows = []
    for clip_id in clip_ids:
        rows.append({
            "model": model,
            "clip_id": clip_id,
            "genre": genres.get(clip_id, ""),
            "tempo_bpm": tempos.get(clip_id),
            "clap_score": clap_scores.get(clip_id),
            "passt_kld": passt_kld,
        })
    return rows


def write_deliverables(
    clip_rows: Iterable[Mapping[str, object]],
    set_rows: Iterable[Mapping[str, object]],
    results_dir: Path,
) -> dict[str, str]:
    """Escribe ambos CSV en results_dir y devuelve sus rutas."""
    results_dir = Path(results_dir)
    clip_rows = list(clip_rows)
    metrics_all = write_metrics_all(clip_rows, results_dir / "metrics_all.csv")
    # Alias con el nombre que espera el dashboard de benchmark (benchmark_analysis):
    # mismas columnas, distinto nombre histórico entre ramas del proyecto.
    clip_level = write_metrics_all(clip_rows, results_dir / "clip_level_metrics.csv")
    set_level = write_set_level(set_rows, results_dir / "set_level_metrics.csv")
    return {
        "metrics_all_csv": str(metrics_all),
        "clip_level_metrics_csv": str(clip_level),
        "set_level_metrics_csv": str(set_level),
    }
