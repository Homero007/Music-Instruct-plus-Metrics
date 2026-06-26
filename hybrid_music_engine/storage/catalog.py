from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig


def list_token_manifests(config: EngineConfig) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    candidates = [
        *config.tokens_dir.glob("input/*/manifest.json"),
        *config.tokens_dir.glob("output/*/manifest.json"),
        *config.datasets_dir.glob("jamendo/*/processed/*/tokens_manifest.json"),
    ]
    for path in sorted(candidates):
        payload = _read_json(path)
        if not payload:
            continue
        total_files = payload.get("total_files") or len(payload.get("entries", []))
        if total_files <= 0:
            continue
        manifests.append(
            {
                "label": _manifest_label(path, payload),
                "path": str(path),
                "kind": payload.get("kind") or "processed",
                "processing_mode": payload.get("processing_mode") or "quick",
                "intended_model": payload.get("intended_model"),
                "total_files": total_files,
                "created_at": payload.get("created_at"),
                "source": _relative_source(config, path),
                "zip_path": payload.get("zip_path"),
                "zip_download_url": data_file_url(config, Path(str(payload.get("zip_path"))))
                if payload.get("zip_path") and Path(str(payload.get("zip_path"))).is_file()
                else None,
            }
        )
    return manifests


def list_token_models(config: EngineConfig) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for path in sorted((config.data_dir / "models" / "tokens").glob("*/model.json")):
        payload = _read_json(path)
        if not payload:
            continue
        model_type = payload.get("model_type") or "markov"
        models.append(
            {
                "label": (
                    f"{payload.get('model_name', 'modelo')} · {model_type} · "
                    f"{payload.get('model_id', path.parent.name)}"
                ),
                "path": str(path),
                "model_id": payload.get("model_id", path.parent.name),
                "model_type": model_type,
                "model_name": payload.get("model_name"),
                "created_at": payload.get("created_at"),
                "order": payload.get("order"),
                "epochs": payload.get("epochs"),
                "vocab_size": payload.get("vocab_size"),
                "token_files": payload.get("token_files"),
                "states": payload.get("states"),
            }
        )
    return models


def list_generations(config: EngineConfig) -> list[dict[str, Any]]:
    generations: list[dict[str, Any]] = []
    for path in sorted((config.data_dir / "generated").glob("*/tokens.json")):
        payload = _read_json(path)
        if not payload:
            continue
        midi_path = Path(str(payload.get("midi_path", "")))
        source_info = _generation_source_info(payload, config)
        layer_midis = {
            name: {
                "path": str(layer_path),
                "download_url": data_file_url(config, layer_path) if layer_path.is_file() else None,
            }
            for name, path_value in (payload.get("layer_midis") or {}).items()
            for layer_path in [Path(str(path_value))]
        }
        generations.append(
            {
                "label": (
                    f"{source_info['label']} · {payload.get('generation_id', path.parent.name)} · "
                    f"{payload.get('duration_seconds_requested', '?')}s"
                ),
                "path": str(path),
                "generation_id": payload.get("generation_id", path.parent.name),
                "generation_mode": source_info["mode"],
                "generation_mode_label": source_info["label"],
                "embedding_path": payload.get("embedding_path"),
                "created_at": payload.get("created_at"),
                "duration_seconds_requested": payload.get("duration_seconds_requested"),
                "seed": payload.get("seed"),
                "condition_genre": payload.get("condition_genre"),
                "feature_tokens": payload.get("feature_tokens", []),
                "token_count": payload.get("token_count"),
                "midi_path": str(midi_path) if midi_path else None,
                "midi_download_url": data_file_url(config, midi_path)
                if midi_path.is_file()
                else None,
                "tokens_download_url": data_file_url(config, path) if path.is_file() else None,
                "layer_midis": layer_midis,
            }
        )
    return generations


def list_rankings(config: EngineConfig) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for ranked_dir in sorted((config.data_dir / "ranked").glob("*")):
        if not ranked_dir.is_dir():
            continue
        reranked_path = ranked_dir / "ranking_reranked.json"
        path = reranked_path if reranked_path.exists() else ranked_dir / "ranking.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        if not payload:
            continue
        source_info = _generation_source_info(payload, config)
        best = next(
            (item for item in payload.get("candidates", []) if item.get("rank") == 1),
            None,
        )
        best_midi = Path(str(best.get("midi_path", ""))) if best else Path("")
        candidates = []
        for item in payload.get("candidates", []):
            midi_path = Path(str(item.get("midi_path", "")))
            render = item.get("render") or {}
            wav_path = Path(str(render.get("wav_path", "")))
            mp3_path = Path(str(render.get("mp3_path", "")))
            score_val = item.get("final_score") if item.get("final_score") is not None else item.get("score")
            candidates.append(
                {
                    "candidate_id": item.get("candidate_id"),
                    "rank": item.get("rank"),
                    "score": score_val,
                    "seed": item.get("seed"),
                    "midi_path": str(midi_path) if midi_path else None,
                    "tokens_path": item.get("tokens_path"),
                    "midi_download_url": data_file_url(config, midi_path)
                    if midi_path.is_file()
                    else None,
                    "wav_download_url": data_file_url(config, wav_path)
                    if wav_path.is_file()
                    else None,
                    "mp3_download_url": data_file_url(config, mp3_path)
                    if mp3_path.is_file()
                    else None,
                    "note_count": (item.get("metrics") or {}).get("note_count"),
                    "duration_seconds": (item.get("metrics") or {}).get("duration_seconds"),
                }
            )
        is_reranked = " (Reranked)" if reranked_path.exists() else ""
        rankings.append(
            {
                "label": (
                    f"{source_info['label']}{is_reranked} · {payload.get('ranked_id', path.parent.name)} · "
                    f"best {payload.get('best_score', 0)}"
                ),
                "path": str(path),
                "ranked_id": payload.get("ranked_id", path.parent.name),
                "generation_mode": source_info["mode"],
                "generation_mode_label": source_info["label"],
                "embedding_path": payload.get("embedding_path"),
                "created_at": payload.get("created_at"),
                "candidates_requested": payload.get("candidates_requested"),
                "condition_genre": payload.get("condition_genre"),
                "best_candidate_id": payload.get("best_candidate_id"),
                "best_score": payload.get("best_score"),
                "best_midi_path": str(best_midi) if best_midi else None,
                "best_midi_download_url": data_file_url(config, best_midi)
                if best_midi.is_file()
                else None,
                "ranking_download_url": data_file_url(config, path) if path.is_file() else None,
                "candidates": candidates,
            }
        )
    return rankings


def list_renders(config: EngineConfig) -> list[dict[str, Any]]:
    renders: list[dict[str, Any]] = []
    render_root = config.data_dir / "renders"
    metadata_paths = sorted(render_root.glob("**/render.json"))
    render_dirs = {path.parent for path in metadata_paths}
    render_dirs.update(path for path in render_root.glob("*") if path.is_dir())
    source_index = _generation_source_index(config)
    for render_dir in sorted(render_dirs):
        metadata_path = render_dir / "render.json"
        payload = _read_json(metadata_path) if metadata_path.exists() else {}
        wavs = sorted(render_dir.glob("*.wav"))
        mp3s = sorted(render_dir.glob("*.mp3"))
        if not payload and not wavs and not mp3s:
            continue
        wav_path = Path(str(payload.get("wav_path", wavs[0] if wavs else "")))
        mp3_path = Path(str(payload.get("mp3_path", mp3s[0] if mp3s else "")))
        source_info = _render_source_info(payload, render_dir, source_index)
        renders.append(
            {
                "label": str(render_dir.relative_to(render_root)),
                "path": str(metadata_path) if metadata_path.exists() else str(render_dir),
                "generation_mode": source_info["mode"],
                "generation_mode_label": source_info["label"],
                "source_generation_id": source_info.get("generation_id"),
                "source_ranked_id": source_info.get("ranked_id"),
                "source_candidate_id": source_info.get("candidate_id"),
                "created_at": payload.get("created_at"),
                "engine": payload.get("engine"),
                "requested_engine": payload.get("requested_engine"),
                "source_midi": payload.get("source_midi"),
                "wav_path": str(wav_path) if wav_path else None,
                "mp3_path": str(mp3_path) if mp3_path else None,
                "wav_download_url": data_file_url(config, wav_path) if wav_path.is_file() else None,
                "mp3_download_url": data_file_url(config, mp3_path) if mp3_path.is_file() else None,
            }
        )
    return renders


def list_blends(config: EngineConfig) -> list[dict[str, Any]]:
    blends: list[dict[str, Any]] = []
    for path in sorted((config.data_dir / "embeddings" / "blends").glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        if payload.get("schema_version") == "weighted-latent-blend-v1":
            source_labels = ", ".join(
                f"{source.get('label')} {source.get('normalized_weight')}"
                for source in payload.get("sources", [])[:3]
            )
            blend_type = "genre_fusion"
            label = f"Fusión de géneros · {payload.get('output_name', 'fusion')} · {source_labels}"
        else:
            blend_type = "pair_blend"
            label = f"Fusión A/B · {payload.get('output_name', 'blend')} · alpha {payload.get('alpha')}"
        blends.append(
            {
                "label": label,
                "path": str(path),
                "blend_id": payload.get("blend_id", path.stem),
                "blend_type": blend_type,
                "created_at": payload.get("created_at"),
                "alpha": payload.get("alpha"),
                "sources": payload.get("sources", []),
                "latent_dim": payload.get("latent_dim"),
            }
        )
    return blends


def list_fusion_comparisons(config: EngineConfig) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for path in sorted((config.data_dir / "fusion_comparisons").glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        best = payload.get("best_fusion") or {}
        comparisons.append(
            {
                "label": f"{payload.get('comparison_id', path.stem)} · best {best.get('label', 'n/a')}",
                "path": str(path),
                "comparison_id": payload.get("comparison_id", path.stem),
                "best_fusion": best,
                "duration_seconds": payload.get("duration_seconds"),
                "candidates_per_fusion": payload.get("candidates_per_fusion"),
                "results": payload.get("results", []),
            }
        )
    return comparisons


def list_token_vae_assets(config: EngineConfig) -> dict[str, list[dict[str, Any]]]:
    root = config.data_dir / "embeddings" / "token_vae"
    models = []
    for path in sorted(root.glob("*/metadata.json")):
        payload = _read_json(path)
        if not payload:
            continue
        models.append(
            {
                "label": f"{payload.get('model_id', path.parent.name)} · {payload.get('latent_dim')}d",
                "path": payload.get("model_path"),
                "metadata_path": str(path),
                "model_id": payload.get("model_id", path.parent.name),
                "latent_dim": payload.get("latent_dim"),
                "sequence_count": payload.get("sequence_count"),
                "created_at": payload.get("created_at"),
            }
        )
    embeddings = []
    for path in sorted((root / "encoded").glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        embeddings.append(
            {
                "label": f"{payload.get('embedding_id', path.stem)} · {payload.get('latent_dim')}d",
                "path": str(path),
                "embedding_id": payload.get("embedding_id", path.stem),
                "model_path": payload.get("model_path"),
                "token_count": payload.get("token_count"),
                "created_at": payload.get("created_at"),
            }
        )
    genre_embeddings = []
    for path in sorted((root / "genres").glob("*/genre_embeddings.json")):
        payload = _read_json(path)
        if not payload:
            continue
        genre_embeddings.append(
            {
                "label": f"{payload.get('output_name', path.parent.name)} · {payload.get('genre_count', 0)} géneros",
                "path": str(path),
                "run_id": payload.get("run_id", path.parent.name),
                "model_path": payload.get("model_path"),
                "genres": payload.get("genres", []),
                "embeddings": payload.get("embeddings", []),
                "created_at": payload.get("created_at"),
            }
        )
    return {"models": models, "embeddings": embeddings, "genre_embeddings": genre_embeddings}


def data_file_url(config: EngineConfig, path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    return f"/api/files?path={resolved}"


def _generation_source_info(payload: dict[str, Any], config: EngineConfig) -> dict[str, str]:
    embedding_path = payload.get("embedding_path")
    if embedding_path:
        path = Path(str(embedding_path)).expanduser()
        embedding_payload = _read_json(path)
        schema = embedding_payload.get("schema_version")
        if schema == "weighted-latent-blend-v1":
            return {"mode": "genre_fusion", "label": "Fusión explícita de géneros"}
        if schema == "latent-blend-v1":
            return {"mode": "pair_blend", "label": "Fusión A/B de embeddings"}
        if schema == "token-vae-genre-embedding-v1":
            genre = embedding_payload.get("genre") or "género"
            return {"mode": "genre_embedding", "label": f"Embedding de género · {genre}"}
        if schema == "token-vae-embedding-v1":
            return {"mode": "token_vae", "label": "Embedding Token-VAE"}
        try:
            relative = path.resolve().relative_to((config.data_dir / "embeddings" / "blends").resolve())
            if relative.parts:
                return {"mode": "blend", "label": "Fusión"}
        except ValueError:
            pass
        return {"mode": "embedding", "label": "Embedding"}
    if payload.get("condition_genre"):
        return {"mode": "genre_condition", "label": f"Transformer por género · {payload.get('condition_genre')}"}
    return {"mode": "transformer", "label": "Transformer normal"}


def _generation_source_index(config: EngineConfig) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for path in sorted((config.data_dir / "generated").glob("*/tokens.json")):
        payload = _read_json(path)
        if not payload:
            continue
        source_info = {
            **_generation_source_info(payload, config),
            "generation_id": str(payload.get("generation_id", path.parent.name)),
        }
        _index_midi_path(index, payload.get("midi_path"), source_info)
        for layer_path in (payload.get("layer_midis") or {}).values():
            _index_midi_path(index, layer_path, source_info)

    for path in sorted((config.data_dir / "ranked").glob("*/ranking.json")):
        payload = _read_json(path)
        if not payload:
            continue
        source_info = {
            **_generation_source_info(payload, config),
            "ranked_id": str(payload.get("ranked_id", path.parent.name)),
        }
        best_id = payload.get("best_candidate_id")
        for item in payload.get("candidates", []):
            candidate_info = {
                **source_info,
                "candidate_id": str(item.get("candidate_id") or ""),
            }
            if item.get("candidate_id") == best_id:
                candidate_info["best_candidate"] = "true"
            _index_midi_path(index, item.get("midi_path"), candidate_info)
            generation = item.get("generation") or {}
            _index_midi_path(index, generation.get("midi_path"), candidate_info)
            for layer_path in (generation.get("layer_midis") or {}).values():
                _index_midi_path(index, layer_path, candidate_info)
    return index


def _render_source_info(
    payload: dict[str, Any],
    render_dir: Path,
    source_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    source_midi = payload.get("source_midi")
    indexed = _lookup_midi_path(source_index, source_midi)
    if indexed:
        return indexed
    render_name = str(render_dir)
    for info in source_index.values():
        ranked_id = info.get("ranked_id")
        candidate_id = info.get("candidate_id")
        if ranked_id and ranked_id in render_name:
            if not candidate_id or candidate_id in render_name:
                return info
    return {"mode": "render", "label": "Render de audio"}


def _index_midi_path(index: dict[str, dict[str, str]], path_value: Any, info: dict[str, str]) -> None:
    key = _normalized_path_key(path_value)
    if key:
        index[key] = info


def _lookup_midi_path(index: dict[str, dict[str, str]], path_value: Any) -> dict[str, str] | None:
    key = _normalized_path_key(path_value)
    return index.get(key) if key else None


def _normalized_path_key(path_value: Any) -> str | None:
    if not path_value:
        return None
    try:
        return str(Path(str(path_value)).expanduser().resolve())
    except OSError:
        return str(path_value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _manifest_label(path: Path, payload: dict[str, Any]) -> str:
    name = payload.get("token_set_name") or payload.get("export_name") or path.parent.name
    total = payload.get("total_files") or len(payload.get("entries", []))
    mode = payload.get("processing_mode")
    prefix = "Token-VAE Demucs" if mode == "token_vae_demucs" else "Rápido"
    return f"{prefix} · {name} · {total} archivos"


def _relative_source(config: EngineConfig, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.data_dir.resolve()))
    except ValueError:
        return str(path)
