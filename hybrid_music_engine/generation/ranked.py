from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
from hybrid_music_engine.render.pedalboard_engine import render_midi_audio
from hybrid_music_engine.tokens.generative_model import generate_tokens_from_model


def generate_ranked_candidates(
    config: EngineConfig,
    *,
    model_path: Path,
    duration_seconds: float,
    output_name: str = "ranked_generation",
    candidates: int = 6,
    seed: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    condition_genre: str | None = None,
    feature_tokens: list[str] | None = None,
    embedding_path: Path | None = None,
    export_layers: bool = True,
    render_best: bool = False,
    render_engine: str = "auto",
    soundfont_path: Path | None = None,
    export_mp3: bool = False,
) -> dict[str, Any]:
    if candidates < 1:
        raise RuntimeError("candidates debe ser mayor o igual a 1.")
    run_id = create_id(output_name, prefix="ranked")
    ranked_dir = config.data_dir / "ranked" / run_id
    ranked_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index in range(1, candidates + 1):
        candidate_seed = (seed + index - 1) if seed is not None else None
        candidate_name = f"{output_name}_candidate_{index:02d}"
        generation = generate_tokens_from_model(
            config,
            model_path=model_path,
            duration_seconds=duration_seconds,
            output_name=candidate_name,
            seed=candidate_seed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            condition_genre=condition_genre,
            feature_tokens=feature_tokens,
            embedding_path=embedding_path,
            export_layers=export_layers,
        )
        midi_path = Path(str(generation["midi_path"]))
        metrics = generation.get("metrics") or analyze_midi_quality(midi_path)
        rows.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "rank": None,
                "seed": candidate_seed,
                "generation": generation,
                "midi_path": str(midi_path),
                "tokens_path": generation.get("path"),
                "layer_midis": generation.get("layer_midis", {}),
                "metrics": metrics,
                "score": float(metrics.get("quality_score", 0.0)),
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best = rows[0]
    best_render = None
    if render_best:
        for row in rows:
            render = render_midi_audio(
                Path(str(row["midi_path"])),
                config.data_dir / "renders" / run_id / str(row["candidate_id"]),
                config=config,
                output_name=str(row["candidate_id"]),
                engine=render_engine,
                soundfont_path=soundfont_path,
                export_mp3=export_mp3,
            )
            row["render"] = render
            if row["candidate_id"] == best["candidate_id"]:
                best_render = render

    summary_path = ranked_dir / "ranking.json"
    summary = {
        "schema_version": "ranked-generation-v1",
        "ranked_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "duration_seconds": duration_seconds,
        "output_name": output_name,
        "candidates_requested": candidates,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "condition_genre": condition_genre,
        "feature_tokens": feature_tokens or [],
        "embedding_path": str(Path(embedding_path).expanduser().resolve()) if embedding_path else None,
        "export_layers": export_layers,
        "best_candidate_id": best["candidate_id"],
        "best_score": best["score"],
        "best_render": best_render,
        "candidates": rows,
        "path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Reranking automático si hay un reward model disponible
    try:
        import os
        from hybrid_music_engine.reward_model.score import RewardScorer
        from hybrid_music_engine.reward_model.rerank import rerank_ranking_file
        reward_dir = os.environ.get("HYBRID_REWARD_DIR", "data/models/reward/v1")
        reward_path = Path(reward_dir)
        if not reward_path.is_absolute():
            reward_path = config.project_root / reward_path
        model_pt = reward_path / "reward_model.pt"
        schema_json = reward_path / "schema.json"
        if model_pt.exists() and schema_json.exists():
            scorer = RewardScorer(model_pt, schema_json)
            rerank_ranking_file(summary_path, scorer, alpha=1.0)
    except Exception as exc:
        import logging
        logging.getLogger("hybrid_music_engine").warning(
            f"Reranking automático omitido o fallido: {exc}"
        )
# Métricas nuevas automáticas (gráficos), DESPUÉS del render.
    try:
        from hybrid_music_engine.new_metrics.integration import run_after_render
        run_after_render(config, run_id, condition_genre=condition_genre)
    except BaseException as exc:
        import logging
        logging.getLogger("hybrid_music_engine").warning(
            f"new_metrics automático omitido o fallido: {exc}"
        )

    return summary
