#!/usr/bin/env python3
"""benchmark_analysis.py — Productos verificables de la Fase 5 del benchmark.

A partir de los CSV de metricas del banco de 100 prompts de MusicCaps produce
los tres productos de "Resultados Esperados":

  1. Tabla comparativa con FAD, CLAP-score, KLD y KAD para los 4 modelos.
  2. Analisis estadistico (Kruskal-Wallis + Dunn con Bonferroni) sobre el
     CLAP-score, global y desglosado por genero musical.
  3. Grafica de radar con las 4 metricas normalizadas a [0, 1] (1 = mejor).

Entradas (esquema del PDF)
--------------------------
  set_level_metrics.csv : model, fad_vggish, fad_pann, mean_clap, std_clap,
                          pct_clap_above_025, kld, kad
  clip_level_metrics.csv: model, clip_id, genre, tempo_bpm, clap_score, passt_kld

Solo depende de numpy, scipy y matplotlib (Dunn implementado a mano).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# Metricas de la tabla comparativa y su direccion (mejor = min/max).
TABLE_METRICS: dict[str, str] = {"fad": "min", "clap": "max", "kld": "min", "kad": "min"}
METRIC_LABELS = {"fad": "FAD", "clap": "CLAP-score", "kld": "KLD", "kad": "KAD"}
# Columna del set-level CSV que alimenta cada metrica de la tabla.
SET_COLUMN = {"fad": "fad_vggish", "clap": "mean_clap", "kld": "kld", "kad": "kad"}


# ── E/S de CSV (sin pandas) ───────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ordered_models(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        model = str(row.get("model", "")).strip()
        if model and model not in seen:
            seen.append(model)
    return seen


# ── Producto 1: tabla comparativa ─────────────────────────────────────────────

def compute_table(set_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Calcula la tabla comparativa y los rangos (sin escribir archivos)."""
    models = _ordered_models(set_rows)
    by_model = {str(row["model"]).strip(): row for row in set_rows}

    # Valores por metrica para rankear.
    values: dict[str, dict[str, float]] = {metric: {} for metric in TABLE_METRICS}
    for model in models:
        row = by_model[model]
        for metric, column in SET_COLUMN.items():
            values[metric][model] = _to_float(row.get(column))

    ranks: dict[str, dict[str, float]] = {}
    for metric, direction in TABLE_METRICS.items():
        ordered = sorted(
            models,
            key=lambda m: (np.inf if np.isnan(values[metric][m]) else values[metric][m]),
            reverse=(direction == "max"),
        )
        ranks[metric] = {model: position + 1 for position, model in enumerate(ordered)}

    table_rows: list[dict[str, Any]] = []
    for model in models:
        row = by_model[model]
        mean_clap = _to_float(row.get("mean_clap"))
        std_clap = _to_float(row.get("std_clap"))
        overall = float(np.mean([ranks[metric][model] for metric in TABLE_METRICS]))
        table_rows.append(
            {
                "model": model,
                "fad_vggish": _to_float(row.get("fad_vggish")),
                "fad_pann": _to_float(row.get("fad_pann")),
                "clap_mean": mean_clap,
                "clap_std": std_clap,
                "pct_clap_above_025": _to_float(row.get("pct_clap_above_025")),
                "kld": _to_float(row.get("kld")),
                "kad": _to_float(row.get("kad")),
                "rank_fad": ranks["fad"][model],
                "rank_clap": ranks["clap"][model],
                "rank_kld": ranks["kld"][model],
                "rank_kad": ranks["kad"][model],
                "overall_rank": overall,
            }
        )
    table_rows.sort(key=lambda item: item["overall_rank"])
    return {
        "rows": table_rows,
        "models": models,
        "methodological_notes": {
            "fad": (
                "FAD (Fréchet Audio Distance) es un escalar de conjunto: se computa "
                "comparando las distribuciones de embeddings de TODO el conjunto generado "
                "contra el de referencia. No existen 100 observaciones independientes de FAD; "
                "no se aplica Kruskal-Wallis."
            ),
            "kld": (
                "KLD de conjunto (set-level) es un único escalar por modelo, derivado de "
                "la divergencia entre distribuciones de embeddings agregadas. "
                "Para pruebas por clip usar 'passt_kld' del CSV clip-level."
            ),
            "kad": (
                "KAD (Kernel Audio Distance / MMD-RBF) es un escalar de conjunto. "
                "No puede tratarse como distribución de 100 observaciones independientes."
            ),
            "clap": (
                "CLAP-score existe por clip → es válido aplicar Kruskal-Wallis y Dunn. "
                "Ver sección 'stats' para IC 95 %, η² y corrección de Bonferroni."
            ),
            "independence": (
                "Los 100 prompts de MusicCaps están distribuidos por género (classical, "
                "electronic, reggaeton, other). Si los audios reales comparten fuente, "
                "estilo, anotador o fueron generados en lotes, la independencia muestral "
                "podría estar comprometida. Reportar los resultados por género (via "
                "'by_genre') permite detectar si el ranking de modelos depende del género."
            ),
        },
    }


def build_comparative_table(set_rows: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    computed = compute_table(set_rows)
    table_rows = computed["rows"]
    models = computed["models"]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "FAD (VGGish)", "FAD (PANN)", "CLAP mean", "CLAP std",
             "% CLAP>0.25", "KLD", "KAD", "rank promedio"]
        )
        for row in table_rows:
            writer.writerow(
                [row["model"], f"{row['fad_vggish']:.3f}", f"{row['fad_pann']:.3f}",
                 f"{row['clap_mean']:.4f}", f"{row['clap_std']:.4f}",
                 f"{row['pct_clap_above_025']:.1f}", f"{row['kld']:.4f}",
                 f"{row['kad']:.4f}", f"{row['overall_rank']:.2f}"]
            )

    md_path = out_dir / "comparison_table.md"
    md = [
        "# Tabla comparativa de modelos (100 prompts de MusicCaps)", "",
        "| Modelo | FAD (VGGish) ↓ | CLAP media ± σ ↑ | CLAP mediana | KLD ↓ | KAD ↓ | Rango prom. |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        md.append(
            f"| {row['model']} "
            f"| {row['fad_vggish']:.3f} "
            f"| {row['clap_mean']:.3f} ± {row['clap_std']:.3f} "
            f"| — "  # mediana disponible en clap_kruskal_dunn.json (nivel clip)
            f"| {row['kld']:.3f} "
            f"| {row['kad']:.3f} "
            f"| {row['overall_rank']:.2f} |"
        )
    md += [
        "",
        "↓ menor es mejor · ↑ mayor es mejor.",
        "El rango promedio combina FAD, CLAP, KLD y KAD (1 = mejor en cada métrica).",
        "",
        "## Notas metodológicas",
        "",
        "- **FAD / KLD (conjunto) / KAD** son escalares de conjunto (un valor por modelo). "
        "No existen 100 observaciones independientes → no se aplica Kruskal-Wallis.",
        "- **CLAP-score** existe por clip → pruebas no-paramétricas válidas "
        "(ver `clap_kruskal_dunn.json` para IC 95 %, η² y Dunn-Bonferroni).",
        "- **KLD por clip** (columna `passt_kld` del CSV clip-level) también se somete "
        "a Kruskal-Wallis como métrica clip-level separada.",
        "- **Independencia muestral**: los 100 prompts están estratificados por género. "
        "Si los audios comparten fuente, anotador o generación por lotes, la independencia "
        "podría estar comprometida. Usar el análisis `by_genre` para verificar estabilidad del ranking.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {"rows": table_rows, "models": models, "csv": str(csv_path), "md": str(md_path)}


# ── Producto 2: Kruskal-Wallis + Dunn (Bonferroni) sobre CLAP ─────────────────

def _dunn_test(groups: list[np.ndarray], labels: list[str]) -> list[dict[str, Any]]:
    """Prueba post-hoc de Dunn con correccion de empates y Bonferroni.

    Devuelve una lista de comparaciones por pares con z, p y p ajustada.
    """
    from scipy.stats import rankdata

    sizes = [len(g) for g in groups]
    pooled = np.concatenate(groups)
    n_total = int(pooled.size)
    ranks = rankdata(pooled, method="average")

    # Rango medio por grupo.
    mean_ranks: list[float] = []
    offset = 0
    for size in sizes:
        mean_ranks.append(float(np.mean(ranks[offset:offset + size])))
        offset += size

    # Correccion por empates: sum(t^3 - t) sobre grupos de valores empatados.
    _, counts = np.unique(pooled, return_counts=True)
    tie_sum = float(np.sum(counts.astype(float) ** 3 - counts.astype(float)))
    tie_term = tie_sum / (12.0 * (n_total - 1)) if n_total > 1 else 0.0
    base_var = n_total * (n_total + 1) / 12.0 - tie_term

    from scipy.stats import norm

    k = len(groups)
    n_comparisons = k * (k - 1) // 2
    comparisons: list[dict[str, Any]] = []
    for i in range(k):
        for j in range(i + 1, k):
            denom = np.sqrt(base_var * (1.0 / sizes[i] + 1.0 / sizes[j]))
            z = (mean_ranks[i] - mean_ranks[j]) / denom if denom > 0 else 0.0
            p = 2.0 * (1.0 - norm.cdf(abs(z)))
            p_adj = min(1.0, p * n_comparisons)
            comparisons.append(
                {
                    "model_a": labels[i],
                    "model_b": labels[j],
                    "z": float(z),
                    "p_value": float(p),
                    "p_bonferroni": float(p_adj),
                    "significant": bool(p_adj < 0.05),
                }
            )
    return comparisons


def _descriptive(arr: np.ndarray) -> dict[str, Any]:
    """Media, std, mediana e IC 95 % para un array de observaciones por clip."""
    from scipy.stats import t as _t

    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "ci_95": (None, None)}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    if n > 1:
        margin = float(_t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
        ci_95: tuple[float | None, float | None] = (mean - margin, mean + margin)
    else:
        ci_95 = (mean, mean)
    return {"n": n, "mean": mean, "std": std, "median": median, "ci_95": ci_95}


def _kruskal_for(metric_by_model: dict[str, list[float]]) -> dict[str, Any]:
    from scipy.stats import kruskal

    labels = [m for m, vals in metric_by_model.items() if len(vals) >= 2]
    groups = [np.asarray(metric_by_model[m], dtype=float) for m in labels]
    descriptive = {m: _descriptive(np.asarray(metric_by_model.get(m, []), dtype=float))
                   for m in metric_by_model}
    if len(groups) < 3:
        return {
            "n_models": len(groups),
            "note": "Kruskal-Wallis requiere >= 3 grupos.",
            "h_statistic": None, "p_value": None, "eta_squared": None,
            "effect_size_label": None, "significant": False,
            "descriptive": descriptive, "dunn": [],
        }
    stat, p_value = kruskal(*groups)
    k = len(labels)
    n_total = sum(g.size for g in groups)
    denom = n_total - k
    eta2 = float(max(0.0, (stat - k + 1) / denom)) if denom > 0 else float("nan")
    if np.isnan(eta2):
        effect_label = "n/a"
    elif eta2 >= 0.14:
        effect_label = "grande"
    elif eta2 >= 0.06:
        effect_label = "mediano"
    else:
        effect_label = "pequeño"
    result: dict[str, Any] = {
        "n_models": k,
        "models": labels,
        "group_sizes": {m: len(metric_by_model[m]) for m in labels},
        "descriptive": descriptive,
        "h_statistic": float(stat),
        "p_value": float(p_value),
        "eta_squared": eta2,
        "effect_size_label": effect_label,
        "significant": bool(p_value < 0.05),
    }
    result["dunn"] = _dunn_test(groups, labels) if p_value < 0.05 else []
    return result


def _collect_clip_metric(
    clip_rows: list[dict[str, str]],
    col: str,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    """Agrupa valores de una columna por modelo y por (género, modelo)."""
    overall: dict[str, list[float]] = {}
    by_genre: dict[str, dict[str, list[float]]] = {}
    for row in clip_rows:
        model = str(row.get("model", "")).strip()
        genre = str(row.get("genre", "")).strip() or "unknown"
        val = _to_float(row.get(col))
        if not model or np.isnan(val):
            continue
        overall.setdefault(model, []).append(val)
        by_genre.setdefault(genre, {}).setdefault(model, []).append(val)
    return overall, by_genre


def compute_clap_stats(clip_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Kruskal-Wallis + Dunn sobre CLAP-score por clip, global y por género."""
    overall, by_genre = _collect_clip_metric(clip_rows, "clap_score")
    return {
        "metric": "clap_score",
        "level": "clip",
        "note": (
            "CLAP-score se mide por clip → las observaciones son independientes "
            "a nivel de muestra y las pruebas no-paramétricas son válidas."
        ),
        "test": "Kruskal-Wallis + Dunn post-hoc (Bonferroni)",
        "alpha": 0.05,
        "overall": _kruskal_for(overall),
        "by_genre": {g: _kruskal_for(v) for g, v in sorted(by_genre.items())},
    }


def compute_passt_kld_stats(clip_rows: list[dict[str, str]]) -> dict[str, Any]:
    """
    Kruskal-Wallis + Dunn sobre KLD estimada por PaSST por clip.

    passt_kld existe a nivel de clip (cada fila del CSV clip-level corresponde
    a un audio generado) → puede tratarse como distribución y someterse a
    pruebas no-paramétricas. Se distingue del KLD de conjunto (set-level),
    que es un único escalar por modelo y no permite este análisis.
    """
    overall, by_genre = _collect_clip_metric(clip_rows, "passt_kld")
    if not any(overall.values()):
        return {
            "metric": "passt_kld",
            "level": "clip",
            "note": "No se encontró la columna 'passt_kld' en el CSV clip-level.",
            "overall": {}, "by_genre": {},
        }
    return {
        "metric": "passt_kld",
        "level": "clip",
        "note": (
            "KLD por clip estimada con PaSST (distribución de probabilidad sobre "
            "clases de audio). Válida para Kruskal-Wallis porque es por muestra. "
            "El KLD de conjunto (set-level) es un único escalar por modelo y "
            "NO puede tratarse como 100 observaciones independientes."
        ),
        "test": "Kruskal-Wallis + Dunn post-hoc (Bonferroni)",
        "alpha": 0.05,
        "overall": _kruskal_for(overall),
        "by_genre": {g: _kruskal_for(v) for g, v in sorted(by_genre.items())},
    }


def clap_kruskal_dunn(clip_rows: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    report = compute_clap_stats(clip_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "clap_kruskal_dunn.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["path"] = str(json_path)
    return report


def passt_kld_kruskal_dunn(clip_rows: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    report = compute_passt_kld_stats(clip_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "passt_kld_stats.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["path"] = str(json_path)
    return report


# ── Producto 3: grafica de radar (4 metricas normalizadas 0-1) ────────────────

def _normalize_metric(values: dict[str, float], direction: str) -> dict[str, float]:
    finite = [v for v in values.values() if not np.isnan(v)]
    if not finite:
        return {model: 0.0 for model in values}
    low, high = min(finite), max(finite)
    span = high - low
    out: dict[str, float] = {}
    for model, value in values.items():
        if np.isnan(value) or span == 0:
            out[model] = 0.5
        elif direction == "min":
            out[model] = (high - value) / span
        else:
            out[model] = (value - low) / span
    return out


def compute_radar(set_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Normaliza las 4 metricas a [0, 1] (1 = mejor) por modelo (sin graficar)."""
    models = _ordered_models(set_rows)
    by_model = {str(row["model"]).strip(): row for row in set_rows}
    raw = {metric: {m: _to_float(by_model[m].get(SET_COLUMN[metric])) for m in models} for metric in TABLE_METRICS}
    normalized = {metric: _normalize_metric(raw[metric], TABLE_METRICS[metric]) for metric in TABLE_METRICS}
    return {
        "models": models,
        "metrics": list(TABLE_METRICS.keys()),
        "metric_labels": {k: METRIC_LABELS[k] for k in TABLE_METRICS},
        "directions": dict(TABLE_METRICS),
        "raw": raw,
        "normalized": normalized,
    }


def radar_chart(set_rows: list[dict[str, str]], out_png: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    computed = compute_radar(set_rows)
    models = computed["models"]
    raw = computed["raw"]
    normalized = computed["normalized"]

    metrics = list(TABLE_METRICS.keys())
    axis_labels = [f"{METRIC_LABELS[m]}\n({'↓' if TABLE_METRICS[m]=='min' else '↑'})" for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    palette = ["#2563EB", "#DC2626", "#059669", "#CA8A04", "#7C3AED"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    for index, model in enumerate(models):
        scores = [normalized[metric][model] for metric in metrics]
        scores += scores[:1]
        color = palette[index % len(palette)]
        ax.plot(angles, scores, color=color, linewidth=2, label=model)
        ax.fill(angles, scores, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8, color="#666")
    ax.set_title("Comparación de modelos — 4 métricas normalizadas (1 = mejor)", pad=24, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out_png), "normalized": normalized, "raw": raw}


# ── Orquestacion ──────────────────────────────────────────────────────────────

def run_analysis(set_csv: Path, clip_csv: Path, out_dir: Path) -> dict[str, Any]:
    set_rows = read_csv(set_csv)
    clip_rows = read_csv(clip_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_comparative_table(set_rows, out_dir)
    stats = clap_kruskal_dunn(clip_rows, out_dir)
    passt_kld = compute_passt_kld_stats(clip_rows)
    (out_dir / "passt_kld_stats.json").write_text(
        json.dumps(passt_kld, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    radar = radar_chart(set_rows, out_dir / "radar_models.png")
    return {
        "table": table,
        "stats": stats,
        "passt_kld_stats": passt_kld,
        "radar": radar,
        "out_dir": str(out_dir),
    }


def benchmark_payload(set_csv: Path, clip_csv: Path) -> dict[str, Any]:
    """
    Devuelve los productos de análisis como datos en memoria (para la API/frontend).

    Métricas de conjunto (FAD, KLD-set, KAD): un escalar por modelo, sin distribución.
    Métricas por clip (CLAP, passt_kld): distribución por modelo → Kruskal-Wallis válido.
    """
    set_csv = Path(set_csv)
    clip_csv = Path(clip_csv)
    if not set_csv.exists() or not clip_csv.exists():
        raise FileNotFoundError(
            "Faltan los CSV de métricas (set_level_metrics.csv / clip_level_metrics.csv). "
            "Genera datos con scripts/make_synthetic_results.py o coloca los CSV reales en results/."
        )
    set_rows = read_csv(set_csv)
    clip_rows = read_csv(clip_csv)
    return {
        "table": compute_table(set_rows),
        "stats": compute_clap_stats(clip_rows),
        "passt_kld_stats": compute_passt_kld_stats(clip_rows),
        "radar": compute_radar(set_rows),
        "sources": {"set_level": str(set_csv), "clip_level": str(clip_csv)},
    }
