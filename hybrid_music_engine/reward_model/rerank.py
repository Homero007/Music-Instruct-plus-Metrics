"""
rerank.py — Re-ranker no invasivo del `ranking.json` existente.

Tu pipeline ya escribe `data/ranked/<id>/ranking.json` con una lista de
candidatas y sus métricas. Este módulo:

  1. Lee ese ranking.
  2. Extrae las métricas embebidas, o las recomputa desde el MIDI si no están.
  3. Llama al reward model para asignar `reward_score` a cada candidata.
  4. Escribe `ranking_reranked.json` AL LADO, sin sobreescribir nada.

Decisiones de diseño:
  • NO mutamos el ranking original. El operador conserva el orden heurístico
    para auditar y comparar.
  • Mezcla configurable: el score final puede ser `reward_score` puro, o una
    combinación lineal con el `score` heurístico original (alpha · reward +
    (1-alpha) · heuristic_normalized). Por defecto: 100 % reward.
  • Tolerante a esquemas: detecta `score`, `final_score`, `total_score` y a
    `metrics`, `features`, `midi_metrics`, etc.

No requiere torch DIRECTAMENTE (delega en RewardScorer), pero torch debe estar
instalado para que el scorer funcione.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .score import RewardScorer

log = logging.getLogger(__name__)


# ── Detección flexible de campos en el ranking ────────────────────────────────

CANDIDATES_KEYS = ("candidates", "items", "rankings", "results")
METRICS_KEYS = ("metrics", "features", "midi_metrics", "stats")
SCORE_KEYS = ("score", "final_score", "total_score", "rank_score")
MIDI_KEYS = ("midi_path", "midi", "path", "generated_midi")


def _first_present(d: Mapping, keys) -> tuple[str | None, object]:
    for k in keys:
        if k in d:
            return k, d[k]
    return None, None


def _extract_candidates(ranking: Mapping) -> tuple[str, list[dict]]:
    key, cands = _first_present(ranking, CANDIDATES_KEYS)
    if cands is None:
        raise ValueError(
            f"ranking.json no contiene ninguna de las claves esperadas: {CANDIDATES_KEYS}"
        )
    if not isinstance(cands, list):
        raise ValueError(f"'{key}' no es lista")
    return key, [dict(c) for c in cands]


# ── Lógica principal ─────────────────────────────────────────────────────────

@dataclass
class RerankReport:
    ranking_path: Path
    output_path: Path
    n_candidates: int
    n_scored: int
    n_missing_metrics: int
    alpha: float    # peso del reward vs heurística
    spearman_vs_original: float | None


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max a [0,1] con tolerancia a constantes."""
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Correlación de Spearman (sin scipy)."""
    if a.size != b.size or a.size < 2:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom < 1e-12:
        return None
    return float((ra * rb).sum() / denom)


def rerank_ranking_file(
    ranking_path: Path,
    scorer: RewardScorer,
    *,
    alpha: float = 1.0,
    metrics_fn: Callable[[Path], Mapping] | None = None,
    output_path: Path | None = None,
) -> RerankReport:
    """
    Re-rankea `ranking_path` y escribe el resultado a `output_path`
    (por defecto `<dir>/ranking_reranked.json`).

    `alpha`: 1.0 = solo reward; 0.0 = solo heurística; intermedio = mezcla.
    `metrics_fn`: si las candidatas no traen métricas embebidas, esta función
                  (ruta MIDI → dict) se usa para extraerlas en caliente.
    """
    ranking_path = Path(ranking_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    key, candidates = _extract_candidates(ranking)
    n = len(candidates)

    # 1) Recolectar métricas por candidata
    metrics_per_cand: list[Mapping | None] = []
    n_missing = 0
    for cand in candidates:
        _, metrics = _first_present(cand, METRICS_KEYS)
        if metrics is None and metrics_fn is not None:
            _, midi = _first_present(cand, MIDI_KEYS)
            if midi:
                midi_path = (ranking_path.parent / midi).resolve() if not Path(midi).is_absolute() else Path(midi)
                if midi_path.exists():
                    try:
                        metrics = metrics_fn(midi_path)
                    except Exception as exc:    # noqa: BLE001
                        log.warning("metrics_fn falló en %s: %s", midi_path, exc)
        if metrics is None:
            n_missing += 1
        metrics_per_cand.append(metrics)

    # 2) Scoring por reward
    valid_idx = [i for i, m in enumerate(metrics_per_cand) if m is not None]
    reward_scores = np.full(n, np.nan, dtype=np.float32)
    if valid_idx:
        scored = scorer.score_batch([metrics_per_cand[i] for i in valid_idx])
        for i, s in zip(valid_idx, scored):
            reward_scores[i] = float(s)

    # 3) Heurística original (si existe)
    heur_scores = np.full(n, np.nan, dtype=np.float32)
    for i, cand in enumerate(candidates):
        _, sc = _first_present(cand, SCORE_KEYS)
        if isinstance(sc, (int, float)):
            heur_scores[i] = float(sc)

    # 4) Combinación final
    reward_norm = _normalize(np.where(np.isnan(reward_scores), 0.0, reward_scores))
    if np.all(np.isnan(heur_scores)):
        final = reward_norm
        alpha_eff = 1.0
    else:
        heur_norm = _normalize(np.where(np.isnan(heur_scores), np.nanmean(heur_scores), heur_scores))
        alpha_eff = float(np.clip(alpha, 0.0, 1.0))
        final = alpha_eff * reward_norm + (1.0 - alpha_eff) * heur_norm

    # 5) Re-ordenar (estable: descendente por final_score)
    order = np.argsort(-final, kind="stable")
    reranked: list[dict] = []
    for new_rank, idx in enumerate(order, start=1):
        c = dict(candidates[idx])
        c["reward_score"] = float(reward_scores[idx]) if not np.isnan(reward_scores[idx]) else None
        c["heuristic_score_original"] = float(heur_scores[idx]) if not np.isnan(heur_scores[idx]) else None
        c["final_score"] = float(final[idx])
        c["rank"] = new_rank
        c["original_index"] = int(idx)
        reranked.append(c)

    # 6) Diagnóstico: Spearman entre orden nuevo y original
    if not np.all(np.isnan(heur_scores)):
        rho = _spearman(reward_scores[~np.isnan(reward_scores) & ~np.isnan(heur_scores)],
                        heur_scores[~np.isnan(reward_scores) & ~np.isnan(heur_scores)])
    else:
        rho = None

    output_path = output_path or (ranking_path.parent / "ranking_reranked.json")
    output_payload = dict(ranking)
    output_payload[key] = reranked
    if reranked:
        output_payload["best_candidate_id"] = reranked[0].get("candidate_id") or reranked[0].get("name") or "candidate-01"
        output_payload["best_score"] = reranked[0].get("final_score") or reranked[0].get("score")
    output_payload["rerank"] = {
        "source_ranking": str(ranking_path),
        "alpha": alpha_eff,
        "n_candidates": n,
        "n_missing_metrics": n_missing,
        "spearman_vs_original": rho,
        "scorer_schema_dim": scorer.schema.dim,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    return RerankReport(
        ranking_path=ranking_path,
        output_path=output_path,
        n_candidates=n,
        n_scored=int((~np.isnan(reward_scores)).sum()),
        n_missing_metrics=n_missing,
        alpha=alpha_eff,
        spearman_vs_original=rho,
    )


def rerank_directory(
    ranked_root: Path,
    scorer: RewardScorer,
    *,
    alpha: float = 1.0,
    metrics_fn: Callable[[Path], Mapping] | None = None,
) -> list[RerankReport]:
    """Aplica rerank a TODOS los `ranking.json` bajo `data/ranked/`."""
    reports: list[RerankReport] = []
    for ranking_path in sorted(Path(ranked_root).rglob("ranking.json")):
        try:
            reports.append(rerank_ranking_file(ranking_path, scorer, alpha=alpha, metrics_fn=metrics_fn))
        except Exception as exc:    # noqa: BLE001
            log.warning("rerank falló en %s: %s", ranking_path, exc)
    return reports
