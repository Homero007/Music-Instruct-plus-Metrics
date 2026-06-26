"""
api_router.py — Router FastAPI OPCIONAL para integrar el reward model en la API.

Cómo usar (sin tocar tu app principal):

  from instruct_music_engine.reward_model.api_router import router as reward_router
  app.include_router(reward_router, prefix="/api/reward", tags=["reward"])

Expone:
  POST /api/reward/score         body {"metrics": {...}} → {"reward_score": float}
  POST /api/reward/rerank        body {"ranking_path": "...", "alpha": 1.0} → reporte
  POST /api/reward/rerank-all    body {"root": "data/ranked", "alpha": 1.0} → lista

El path del modelo se resuelve por (en orden):
  1. campo `model_dir` del body, si viene
  2. variable de entorno HYBRID_REWARD_DIR
  3. default: data/models/reward/v1

El scorer se cachea en memoria por (model_dir, device) para no recargar pesos.

Si FastAPI no está disponible, este módulo NO se importa por nada del paquete:
es opcional por diseño.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError as exc:    # mantenemos el paquete utilizable sin FastAPI
    raise ImportError(
        "api_router requiere fastapi y pydantic. `pip install fastapi pydantic`"
    ) from exc

from .metrics_provider import make_metrics_fn
from .rerank import rerank_directory, rerank_ranking_file
from .score import RewardScorer


router = APIRouter()

_DEFAULT_DIR = os.environ.get("HYBRID_REWARD_DIR", "data/models/reward/v1")
_scorer_cache: dict[tuple[str, str], RewardScorer] = {}
_cache_lock = Lock()


def _get_scorer(model_dir: str | None, device: str) -> RewardScorer:
    base = Path(model_dir or _DEFAULT_DIR)
    model_path = base / "reward_model.pt"
    schema_path = base / "schema.json"
    if not model_path.exists() or not schema_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Reward model no encontrado en {base}. Entrena primero con el CLI.",
        )
    key = (str(base.resolve()), device)
    with _cache_lock:
        if key not in _scorer_cache:
            _scorer_cache[key] = RewardScorer(model_path, schema_path, device=device)
        return _scorer_cache[key]


# ── Request models ───────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    metrics: dict
    model_dir: str | None = None
    device: str = "cpu"


class RerankRequest(BaseModel):
    ranking_path: str
    alpha: float = 1.0
    output_path: str | None = None
    model_dir: str | None = None
    metrics_mode: str = "auto"
    api_url: str | None = None
    device: str = "cpu"


class RerankAllRequest(BaseModel):
    root: str = "data/ranked"
    alpha: float = 1.0
    model_dir: str | None = None
    metrics_mode: str = "auto"
    api_url: str | None = None
    device: str = "cpu"


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/score")
def score_endpoint(req: ScoreRequest) -> dict:
    scorer = _get_scorer(req.model_dir, req.device)
    return {"reward_score": scorer.score(req.metrics)}


@router.post("/rerank")
def rerank_endpoint(req: RerankRequest) -> dict:
    scorer = _get_scorer(req.model_dir, req.device)
    metrics_fn = make_metrics_fn(req.metrics_mode, api_url=req.api_url or "http://127.0.0.1:8100")
    ranking_path = Path(req.ranking_path)
    if not ranking_path.exists():
        raise HTTPException(status_code=400, detail=f"ranking.json no existe: {ranking_path}")
    report = rerank_ranking_file(
        ranking_path, scorer,
        alpha=req.alpha, metrics_fn=metrics_fn,
        output_path=Path(req.output_path) if req.output_path else None,
    )
    return {
        "ranking_path": str(report.ranking_path),
        "output_path": str(report.output_path),
        "n_candidates": report.n_candidates,
        "n_scored": report.n_scored,
        "n_missing_metrics": report.n_missing_metrics,
        "alpha": report.alpha,
        "spearman_vs_original": report.spearman_vs_original,
    }


@router.post("/rerank-all")
def rerank_all_endpoint(req: RerankAllRequest) -> dict:
    scorer = _get_scorer(req.model_dir, req.device)
    metrics_fn = make_metrics_fn(req.metrics_mode, api_url=req.api_url or "http://127.0.0.1:8100")
    reports = rerank_directory(Path(req.root), scorer, alpha=req.alpha, metrics_fn=metrics_fn)
    return {
        "n_reranked": len(reports),
        "results": [{
            "ranking_path": str(r.ranking_path),
            "output_path": str(r.output_path),
            "n_candidates": r.n_candidates,
            "spearman_vs_original": r.spearman_vs_original,
        } for r in reports],
    }
