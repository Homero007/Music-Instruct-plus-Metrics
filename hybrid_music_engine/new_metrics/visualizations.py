#!/usr/bin/env python3
"""
visualizations.py — Gráficas comparativas de la Fase 5.

Funciones independientes que producen los .png pedidos por el protocolo a partir
de estructuras de datos simples (dicts), sin acoplarse a pandas:

  • radar_chart           : 4 modelos × N métricas normalizadas (1 = mejor).
  • heatmap_by_genre      : CLAP-score medio por modelo × género.
  • scatter_fad_vs_clap   : FAD vs CLAP, un punto por modelo (calidad vs semántica).
  • violin_by_model       : distribución por clip de una métrica (violín o boxplot).

Todas usan backend 'Agg' (sin display) y devuelven la ruta del PNG generado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


_PALETTE = ["#2563EB", "#DC2626", "#059669", "#CA8A04", "#7C3AED", "#0891B2"]


# ── Radar de métricas normalizadas ──────────────────────────────────────────

def radar_chart(
    metrics_by_model: Mapping[str, Mapping[str, float]],
    higher_is_better: Mapping[str, bool],
    out_path: Path,
    *,
    title: str = "Comparación de modelos (métricas normalizadas, 1 = mejor)",
) -> Path:
    """
    metrics_by_model : {modelo: {metrica: valor}}
    higher_is_better : {metrica: True/False}  (dirección de cada métrica)

    Cada métrica se normaliza a [0,1] entre modelos de modo que 1 = el mejor
    modelo en esa métrica (respetando la dirección). Métricas con valor faltante
    se imputan a 0 (peor) para esa esquina.
    """
    plt = _setup()
    models = list(metrics_by_model.keys())
    metrics = list(higher_is_better.keys())
    if not models or not metrics:
        raise ValueError("Se necesitan modelos y métricas para el radar.")

    # Normalización por métrica entre modelos.
    norm: dict[str, dict[str, float]] = {m: {} for m in models}
    for metric in metrics:
        vals = np.array(
            [metrics_by_model[m].get(metric, np.nan) for m in models], dtype=np.float64
        )
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            for m in models:
                norm[m][metric] = 0.0
            continue
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo
        for m, v in zip(models, vals):
            if not np.isfinite(v):
                norm[m][metric] = 0.0
            elif span < 1e-12:
                norm[m][metric] = 1.0
            else:
                scaled = (v - lo) / span
                norm[m][metric] = scaled if higher_is_better[metric] else (1.0 - scaled)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # cerrar el polígono

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for idx, model in enumerate(models):
        values = [norm[model][metric] for metric in metrics]
        values += values[:1]
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Heatmap CLAP por modelo × género ────────────────────────────────────────

def heatmap_by_genre(
    matrix: Mapping[str, Mapping[str, float]],
    out_path: Path,
    *,
    title: str = "CLAP-score medio por modelo × género",
    value_fmt: str = "{:.3f}",
) -> Path:
    """matrix : {modelo: {genero: valor_medio}}."""
    plt = _setup()
    models = list(matrix.keys())
    genres = sorted({g for row in matrix.values() for g in row})
    if not models or not genres:
        raise ValueError("Se necesitan modelos y géneros para el heatmap.")

    data = np.full((len(models), len(genres)), np.nan, dtype=np.float64)
    for i, model in enumerate(models):
        for j, genre in enumerate(genres):
            v = matrix[model].get(genre)
            if v is not None and np.isfinite(v):
                data[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(1.4 * len(genres) + 3, 0.8 * len(models) + 2))
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#E5E7EB")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(genres)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, value_fmt.format(data[i, j]), ha="center", va="center",
                        color="white", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="CLAP-score")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ── Dispersión FAD vs CLAP ──────────────────────────────────────────────────

def scatter_fad_vs_clap(
    points: Mapping[str, tuple[float, float]],
    out_path: Path,
    *,
    title: str = "FAD vs CLAP-score (calidad acústica vs alineación semántica)",
) -> Path:
    """points : {modelo: (fad, clap)}. Anota cada punto con el nombre del modelo."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = [], []
    for idx, (model, (fad, clap)) in enumerate(points.items()):
        if fad is None or clap is None or not (np.isfinite(fad) and np.isfinite(clap)):
            continue
        color = _PALETTE[idx % len(_PALETTE)]
        ax.scatter(fad, clap, s=120, color=color, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(model, (fad, clap), textcoords="offset points", xytext=(8, 6), fontsize=9)
        xs.append(float(fad))
        ys.append(float(clap))

    if len(xs) >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.02, 0.98, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="#F3F4F6", ec="#9CA3AF"))

    ax.set_xlabel("FAD (↓ mejor calidad acústica)")
    ax.set_ylabel("CLAP-score (↑ mejor alineación)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ── Violín / boxplot por modelo ─────────────────────────────────────────────

def violin_by_model(
    groups: Mapping[str, Sequence[float]],
    out_path: Path,
    *,
    metric_name: str = "CLAP-score",
    kind: str = "violin",
) -> Path:
    """groups : {modelo: [valores_por_clip]}. kind='violin' o 'box'."""
    plt = _setup()
    labels, data = [], []
    for model, values in groups.items():
        arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
        if arr.size:
            labels.append(model)
            data.append(arr)
    if not data:
        raise ValueError("No hay datos para el gráfico por modelo.")

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 5))
    if kind == "violin":
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for idx, body in enumerate(parts["bodies"]):
            body.set_facecolor(_PALETTE[idx % len(_PALETTE)])
            body.set_alpha(0.55)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        ax.boxplot(data, labels=labels, showmeans=True)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribución de {metric_name} por modelo")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path
