#!/usr/bin/env python3
"""
stats_tests.py — Pruebas de hipótesis para CLAP-score (y métricas por clip).

Implementa el análisis estadístico de la Fase 5 sin asumir normalidad:

  • Kruskal-Wallis  : ¿hay diferencias entre modelos? (scipy.stats.kruskal)
  • Post-hoc de Dunn: comparaciones por pares si Kruskal-Wallis es significativo,
                      con corrección de Bonferroni. Implementado a mano (solo
                      scipy.stats.norm + numpy) para NO depender de scikit_posthocs.
  • Consistencia por género: ¿el ranking de modelos es estable entre géneros?

Trabaja sobre muestras por clip: groups = {modelo: [valores_por_clip]}.

Uso como módulo:
    from stats_tests import analyze_metric, per_genre_consistency
    result = analyze_metric({"musicgen_small": [...], "audioldm2": [...]})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# ── Kruskal-Wallis ──────────────────────────────────────────────────────────

def kruskal_wallis(groups: Mapping[str, Sequence[float]]) -> dict:
    """
    Prueba de Kruskal-Wallis entre >=2 grupos.

    Devuelve H, p, grados de libertad y tamaño del efecto eta-cuadrado (η²):
        η²_H = (H − k + 1) / (N − k)
    donde k = número de grupos y N = total de observaciones.
    Convenciones de Cohen para η²: 0.01 pequeño, 0.06 mediano, 0.14 grande.
    """
    from scipy.stats import kruskal

    clean = {
        name: np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
        for name, values in groups.items()
    }
    clean = {name: arr for name, arr in clean.items() if arr.size > 0}
    if len(clean) < 2:
        return {
            "test": "kruskal_wallis",
            "h_statistic": float("nan"),
            "p_value": float("nan"),
            "dof": max(0, len(clean) - 1),
            "n_groups": len(clean),
            "eta_squared": float("nan"),
            "effect_size_label": "n/a",
            "significant": False,
            "note": "Kruskal-Wallis requiere al menos 2 grupos con datos.",
        }
    h, p = kruskal(*clean.values())
    k = len(clean)
    n_total = sum(arr.size for arr in clean.values())
    denom = n_total - k
    eta2 = float((h - k + 1) / denom) if denom > 0 else float("nan")
    eta2 = max(0.0, eta2)  # η² no puede ser negativo
    if np.isnan(eta2):
        effect_label = "n/a"
    elif eta2 >= 0.14:
        effect_label = "grande"
    elif eta2 >= 0.06:
        effect_label = "mediano"
    else:
        effect_label = "pequeño"
    return {
        "test": "kruskal_wallis",
        "h_statistic": float(h),
        "p_value": float(p),
        "dof": k - 1,
        "n_groups": k,
        "n_total": n_total,
        "eta_squared": eta2,
        "effect_size_label": effect_label,
        "significant": bool(p < 0.05),
        "note": "Diferencias significativas entre modelos." if p < 0.05
        else "Sin evidencia suficiente de diferencias entre modelos.",
    }


# ── Post-hoc de Dunn con corrección de Bonferroni ───────────────────────────

def dunn_test(groups: Mapping[str, Sequence[float]], p_adjust: str = "bonferroni") -> dict:
    """
    Prueba post-hoc de Dunn (1964) con corrección de empates. Devuelve, por par
    de modelos, el estadístico z y el p-value ajustado (Bonferroni por defecto).

    SE_ij = sqrt( [ N(N+1)/12 − Σ(t³−t)/(12(N−1)) ] · (1/n_i + 1/n_j) )
    z_ij  = (R̄_i − R̄_j) / SE_ij ,  p two-sided desde la normal estándar.
    """
    from scipy.stats import norm, rankdata

    clean = {
        name: np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
        for name, values in groups.items()
    }
    clean = {name: arr for name, arr in clean.items() if arr.size > 0}
    names = list(clean.keys())
    k = len(names)
    if k < 2:
        return {"test": "dunn", "p_adjust": p_adjust, "comparisons": [], "note": "Se requieren >=2 grupos."}

    pooled = np.concatenate([clean[name] for name in names])
    ranks = rankdata(pooled, method="average")
    big_n = pooled.size

    # Rango medio por grupo.
    mean_ranks: dict[str, float] = {}
    sizes: dict[str, int] = {}
    offset = 0
    for name in names:
        n = clean[name].size
        mean_ranks[name] = float(ranks[offset:offset + n].mean())
        sizes[name] = n
        offset += n

    # Corrección por empates: Σ(t³ − t) sobre grupos de valores idénticos.
    _, counts = np.unique(pooled, return_counts=True)
    tie_sum = float(np.sum(counts ** 3 - counts))
    sigma2 = (big_n * (big_n + 1) / 12.0) - tie_sum / (12.0 * (big_n - 1)) if big_n > 1 else 0.0

    n_comparisons = k * (k - 1) // 2
    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = names[i], names[j]
            se = np.sqrt(sigma2 * (1.0 / sizes[a] + 1.0 / sizes[b])) if sigma2 > 0 else 0.0
            z = (mean_ranks[a] - mean_ranks[b]) / se if se > 0 else 0.0
            p_raw = 2.0 * (1.0 - norm.cdf(abs(z)))
            if p_adjust == "bonferroni":
                p_adj = min(1.0, p_raw * n_comparisons)
            else:
                p_adj = p_raw
            comparisons.append({
                "model_a": a,
                "model_b": b,
                "mean_rank_a": mean_ranks[a],
                "mean_rank_b": mean_ranks[b],
                "z": float(z),
                "p_raw": float(p_raw),
                "p_adjusted": float(p_adj),
                "significant": bool(p_adj < 0.05),
            })
    return {
        "test": "dunn",
        "p_adjust": p_adjust,
        "n_comparisons": n_comparisons,
        "mean_ranks": mean_ranks,
        "comparisons": comparisons,
    }


# ── Análisis combinado de una métrica ───────────────────────────────────────

def analyze_metric(
    groups: Mapping[str, Sequence[float]],
    *,
    metric_name: str = "clap_score",
    alpha: float = 0.05,
) -> dict:
    """
    Análisis completo de una métrica por clip entre modelos:
      1. Estadística descriptiva (media, σ, n) por modelo.
      2. Kruskal-Wallis.
      3. Si p < alpha, post-hoc de Dunn con Bonferroni.
    """
    descriptive = {}
    for name, values in groups.items():
        arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean()) if n else None
        std = float(arr.std(ddof=1)) if n > 1 else (0.0 if n == 1 else None)
        median = float(np.median(arr)) if n else None
        # IC 95 % por t de Student (bilateral, α = 0.05)
        if n > 1 and std is not None:
            from scipy.stats import t as _t
            margin = float(_t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
            ci_95 = (float(mean - margin), float(mean + margin))
        else:
            ci_95 = (mean, mean)
        descriptive[name] = {
            "n": n,
            "mean": mean,
            "std": std,
            "median": median,
            "ci_95": ci_95,
        }

    kw = kruskal_wallis(groups)
    result = {
        "metric": metric_name,
        "alpha": alpha,
        "descriptive": descriptive,
        "kruskal_wallis": kw,
        "posthoc_dunn": None,
    }
    if kw.get("significant"):
        result["posthoc_dunn"] = dunn_test(groups, p_adjust="bonferroni")
    return result


# ── Consistencia de ranking por género ──────────────────────────────────────

def per_genre_consistency(
    by_genre: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    higher_is_better: bool = True,
) -> dict:
    """
    Para {género: {modelo: [valores_por_clip]}}, calcula el ranking de modelos
    dentro de cada género (por la media) y mide si es consistente entre géneros
    con la W de Kendall (1 = ranking idéntico en todos los géneros).
    """
    from scipy.stats import rankdata

    genre_means: dict[str, dict[str, float]] = {}
    models: set[str] = set()
    for genre, model_values in by_genre.items():
        means = {}
        for model, values in model_values.items():
            arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
            if arr.size:
                means[model] = float(arr.mean())
                models.add(model)
        if means:
            genre_means[genre] = means

    model_list = sorted(models)
    rankings: dict[str, dict[str, float]] = {}
    rank_matrix = []
    for genre, means in genre_means.items():
        present = [m for m in model_list if m in means]
        if len(present) < 2:
            continue
        vals = np.array([means[m] for m in present], dtype=np.float64)
        # rank 1 = mejor. Para "mayor es mejor" se rankea el negativo.
        ranks = rankdata(-vals if higher_is_better else vals, method="average")
        rankings[genre] = {m: float(r) for m, r in zip(present, ranks)}
        if len(present) == len(model_list):
            rank_matrix.append([rankings[genre][m] for m in model_list])

    kendall_w = None
    if len(rank_matrix) >= 2 and len(model_list) >= 2:
        rm = np.asarray(rank_matrix, dtype=np.float64)  # (n_generos, n_modelos)
        n_judges = rm.shape[0]
        k = rm.shape[1]
        rank_sums = rm.sum(axis=0)
        s = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
        denom = n_judges ** 2 * (k ** 3 - k)
        kendall_w = float(12.0 * s / denom) if denom > 0 else None

    return {
        "models": model_list,
        "genre_means": genre_means,
        "rankings_per_genre": rankings,
        "kendall_w": kendall_w,
        "consistent": bool(kendall_w is not None and kendall_w >= 0.7),
        "note": "W de Kendall sobre rankings por género (1 = ranking idéntico). "
                "W>=0.7 indica un ranking de modelos bastante consistente entre géneros.",
    }


# ── Persistencia ────────────────────────────────────────────────────────────

def save_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
