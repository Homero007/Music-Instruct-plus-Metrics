
from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hybrid_music_engine.audio_classifier.model import (
    AudioCentroidClassifier,
    train_audio_classifier,
)
from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.generation.ranked import generate_ranked_candidates
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
from hybrid_music_engine.storage.catalog import data_file_url, list_rankings, list_renders

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}
MIDI_EXTENSIONS = {".mid", ".midi"}
DEFAULT_REAL_AUDIO_ROOT = Path("data/datasets/jamendo/delivery_jamendo_150/audio")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _audio_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _midi_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in MIDI_EXTENSIONS)


def _default_real_root(config: EngineConfig) -> Path:
    return (config.project_root / DEFAULT_REAL_AUDIO_ROOT).resolve()


def _counts_by_genre(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        p.name: len(_audio_files(p))
        for p in sorted(root.iterdir())
        if p.is_dir() and len(_audio_files(p)) > 0
    }


def _recommended_distribution(counts: dict[str, int], target_total: int = 100) -> dict[str, int]:
    available = {genre: int(count) for genre, count in counts.items() if int(count) > 0}
    if not available:
        return {}
    target = min(int(target_total), sum(available.values()))
    if target <= 0:
        return {}
    total = sum(available.values())
    raw = {genre: target * (count / total) for genre, count in available.items()}
    result = {genre: min(available[genre], int(math.floor(value))) for genre, value in raw.items()}
    remaining = target - sum(result.values())
    order = sorted(available, key=lambda genre: (raw[genre] - math.floor(raw[genre])), reverse=True)
    while remaining > 0:
        moved = False
        for genre in order:
            if result[genre] < available[genre]:
                result[genre] += 1
                remaining -= 1
                moved = True
                if remaining == 0:
                    break
        if not moved:
            break
    return {genre: count for genre, count in result.items() if count > 0}


def evaluation_availability(
    config: EngineConfig,
    *,
    real_audio_root: Path | None = None,
    target_total: int = 100,
) -> dict[str, Any]:
    root = Path(real_audio_root).expanduser().resolve() if real_audio_root else _default_real_root(config)
    counts = _counts_by_genre(root)
    return {
        "real_audio_root": str(root),
        "counts": counts,
        "recommended_distribution": _recommended_distribution(counts, target_total=target_total),
        "max_distribution": counts,
        "target_total": int(target_total),
        "total_available": int(sum(counts.values())),
    }


def _normalize_distribution(
    config: EngineConfig,
    distribution: dict[str, int] | None,
    *,
    real_audio_root: Path,
    target_total: int = 100,
) -> dict[str, int]:
    availability = evaluation_availability(config, real_audio_root=real_audio_root, target_total=target_total)
    counts = availability["counts"]
    requested = distribution or availability["recommended_distribution"]
    if not requested:
        raise RuntimeError(f"No hay audios reales disponibles en {real_audio_root}.")
    normalized: dict[str, int] = {}
    errors: list[str] = []
    for genre, raw_count in requested.items():
        count = int(raw_count)
        if count <= 0:
            continue
        available = int(counts.get(genre, 0))
        if available <= 0:
            errors.append(f"'{genre}' no existe o no tiene audios reales.")
        elif count > available:
            errors.append(f"'{genre}' pidió {count}, pero solo hay {available}.")
        else:
            normalized[genre] = count
    if errors:
        raise RuntimeError("Distribución inválida: " + " ".join(errors))
    if not normalized:
        raise RuntimeError("La distribución no contiene canciones a generar.")
    return normalized


def _copy_file(src: Path, dst: Path) -> str | None:
    if not src or not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _candidate_audio_from_render(render: dict[str, Any] | None) -> tuple[Path | None, Path | None]:
    render = render or {}
    wav = Path(str(render.get("wav_path", ""))) if render.get("wav_path") else None
    mp3 = Path(str(render.get("mp3_path", ""))) if render.get("mp3_path") else None
    return wav, mp3


def _existing_audio_from_candidate(candidate: dict[str, Any]) -> tuple[Path | None, Path | None]:
    wav = Path(str(candidate.get("wav_path", ""))) if candidate.get("wav_path") else None
    mp3 = Path(str(candidate.get("mp3_path", ""))) if candidate.get("mp3_path") else None
    if not wav or not wav.is_file():
        wav_url_path = candidate.get("wav_download_url")
        wav = None if wav_url_path else wav
    if not mp3 or not mp3.is_file():
        mp3_url_path = candidate.get("mp3_download_url")
        mp3 = None if mp3_url_path else mp3
    return wav, mp3


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _candidate_genre(ranking_payload: dict[str, Any], candidate: dict[str, Any]) -> str:
    generation = candidate.get("generation") or {}
    genre = generation.get("condition_genre") or ranking_payload.get("condition_genre")
    if genre:
        return str(genre)
    feature_tokens = generation.get("feature_tokens") or ranking_payload.get("feature_tokens") or []
    for token in feature_tokens:
        if isinstance(token, str) and token.startswith("genre:"):
            return token.split(":", 1)[1] or "unknown"
    if ranking_payload.get("embedding_path"):
        return "fusion"
    return "unknown"


def _candidate_audio_paths(candidate: dict[str, Any]) -> tuple[Path | None, Path | None]:
    render = candidate.get("render") or {}
    wav = Path(str(render.get("wav_path", ""))) if render.get("wav_path") else None
    mp3 = Path(str(render.get("mp3_path", ""))) if render.get("mp3_path") else None
    return (wav if wav and wav.is_file() else None, mp3 if mp3 and mp3.is_file() else None)


def _copy_reference_audio(real_root: Path, copied_real_root: Path, genres: dict[str, int]) -> dict[str, int]:
    copied: dict[str, int] = {}
    available_counts = _counts_by_genre(real_root)
    known_genres = [genre for genre in genres if genre in available_counts]
    if not known_genres:
        known_genres = sorted(available_counts)
    if not known_genres:
        return copied
    for genre in known_genres:
        requested = max(1, int(genres.get(genre, 1)))
        files = _audio_files(real_root / genre)[:requested]
        for index, real_file in enumerate(files, start=1):
            dst_genre = genre
            _copy_file(real_file, copied_real_root / dst_genre / f"real_{index:03d}{real_file.suffix.lower()}")
        copied[genre] = len(files)
    return copied


def _copied_reference_audio(copied_real_root: Path) -> dict[str, list[Path]]:
    if not copied_real_root.exists():
        return {}
    return {
        genre_dir.name: _audio_files(genre_dir)
        for genre_dir in sorted(copied_real_root.iterdir())
        if genre_dir.is_dir()
    }


def _public_file(config: EngineConfig, path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    return data_file_url(config, file_path) if file_path.exists() else None


def _track_audio_path(track: dict[str, Any]) -> Path | None:
    paths = track.get("paths") or {}
    for key in ("wav_path", "mp3_path"):
        value = paths.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    return None


def _make_same_genre_pairs(
    config: EngineConfig,
    *,
    selected_tracks: list[dict[str, Any]],
    copied_real_root: Path,
) -> list[dict[str, Any]]:
    references = _copied_reference_audio(copied_real_root)
    pair_counters: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    for index, track in enumerate(selected_tracks, start=1):
        genre = str(track.get("genre") or "unknown")
        real_files = references.get(genre) or []
        if not real_files:
            pairs.append(
                {
                    "pair_id": f"pair_{index:03d}",
                    "genre": genre,
                    "track_id": track.get("track_id"),
                    "candidate_id": track.get("candidate_id"),
                    "source": track.get("source") or {},
                    "generated": {
                        "audio_path": str(_track_audio_path(track)) if _track_audio_path(track) else None,
                        "audio_url": _public_file(config, str(_track_audio_path(track)) if _track_audio_path(track) else None),
                        "midi_path": (track.get("paths") or {}).get("midi_path"),
                        "midi_url": _public_file(config, (track.get("paths") or {}).get("midi_path")),
                    },
                    "original": None,
                    "error": "No hay audio original copiado para este género.",
                }
            )
            continue
        real_index = pair_counters.get(genre, 0) % len(real_files)
        pair_counters[genre] = pair_counters.get(genre, 0) + 1
        generated_audio = _track_audio_path(track)
        real_audio = real_files[real_index]
        pairs.append(
            {
                "pair_id": f"pair_{index:03d}",
                "genre": genre,
                "track_id": track.get("track_id"),
                "candidate_id": track.get("candidate_id"),
                "source": track.get("source") or {},
                "rank": track.get("rank"),
                "score": track.get("score"),
                "seed": track.get("seed"),
                "generated": {
                    "audio_path": str(generated_audio) if generated_audio else None,
                    "audio_url": _public_file(config, str(generated_audio) if generated_audio else None),
                    "midi_path": (track.get("paths") or {}).get("midi_path"),
                    "midi_url": _public_file(config, (track.get("paths") or {}).get("midi_path")),
                },
                "original": {
                    "audio_path": str(real_audio),
                    "audio_url": _public_file(config, str(real_audio)),
                    "label": real_audio.name,
                },
            }
        )
    return pairs


def _evaluation_tracks_from_manifest_or_metadata(evaluation_dir: Path) -> list[dict[str, Any]]:
    manifest = _safe_load_json(evaluation_dir / "manifest.json")
    tracks = [
        track
        for track in manifest.get("tracks", [])
        if isinstance(track, dict) and track.get("status") == "completed"
    ]
    if tracks:
        return tracks
    generated_root = evaluation_dir / "generated"
    loaded: list[dict[str, Any]] = []
    for metadata_path in sorted(generated_root.glob("*/*/metadata.json")):
        metadata = _safe_load_json(metadata_path)
        if metadata and metadata.get("status") == "completed":
            loaded.append(metadata)
    return loaded


def _ensure_evaluation_pairs(config: EngineConfig, evaluation_dir: Path) -> list[dict[str, Any]]:
    manifest_path = evaluation_dir / "manifest.json"
    manifest = _safe_load_json(manifest_path)
    pairs = manifest.get("pairs") or []
    if pairs:
        return pairs
    tracks = _evaluation_tracks_from_manifest_or_metadata(evaluation_dir)
    if not tracks:
        return []
    pairs = _make_same_genre_pairs(
        config,
        selected_tracks=tracks,
        copied_real_root=evaluation_dir / "real",
    )
    if pairs:
        manifest["pairs"] = pairs
        if manifest_path.exists():
            _json_dump(manifest_path, manifest)
        _json_dump(evaluation_dir / "metrics" / "pair_metrics.json", {"pairs": pairs})
    return pairs


def _source_group(source_info: dict[str, Any]) -> str:
    mode = str(source_info.get("generation_mode") or source_info.get("mode") or "")
    if mode in {"genre_fusion", "pair_blend", "blend"}:
        return "fusion"
    if mode == "render":
        return "renders"
    return "normal"


def _empty_source_groups() -> dict[str, dict[str, Any]]:
    return {
        "normal": {
            "label": "Generación normal",
            "sources": [],
            "total_audio_ready": 0,
            "total_missing_audio": 0,
        },
        "fusion": {
            "label": "Fusión explícita de géneros",
            "sources": [],
            "total_audio_ready": 0,
            "total_missing_audio": 0,
        },
        "renders": {
            "label": "Renders sueltos",
            "sources": [],
            "total_audio_ready": 0,
            "total_missing_audio": 0,
        },
    }


def _source_genre_summary(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    genres: dict[str, dict[str, Any]] = {}
    for candidate in source.get("candidates") or []:
        genre = str(candidate.get("genre") or "unknown")
        bucket = genres.setdefault(
            genre,
            {
                "genre": genre,
                "audio_ready_count": 0,
                "missing_audio_count": 0,
                "total_candidates": 0,
                "max_selectable": 0,
                "candidates": [],
            },
        )
        bucket["total_candidates"] += 1
        if candidate.get("audio_ready"):
            bucket["audio_ready_count"] += 1
            bucket["max_selectable"] += 1
        else:
            bucket["missing_audio_count"] += 1
        bucket["candidates"].append(candidate)
    return genres


def _add_source_to_genre_groups(genre_groups: dict[str, dict[str, Any]], source: dict[str, Any]) -> None:
    for genre, summary in (source.get("genres") or {}).items():
        group = genre_groups.setdefault(
            genre,
            {
                "genre": genre,
                "label": genre,
                "sources": [],
                "total_audio_ready": 0,
                "total_missing_audio": 0,
                "total_candidates": 0,
            },
        )
        source_row = {
            "source_type": source.get("source_type"),
            "source_id": source.get("source_id"),
            "label": source.get("label"),
            "created_at": source.get("created_at"),
            "generation_mode": source.get("generation_mode"),
            "generation_mode_label": source.get("generation_mode_label"),
            "source_group": source.get("source_group"),
            "audio_ready_count": summary.get("audio_ready_count", 0),
            "missing_audio_count": summary.get("missing_audio_count", 0),
            "total_candidates": summary.get("total_candidates", 0),
            "max_selectable": summary.get("max_selectable", 0),
            "candidates": summary.get("candidates", []),
        }
        group["sources"].append(source_row)
        group["total_audio_ready"] += int(summary.get("audio_ready_count", 0))
        group["total_missing_audio"] += int(summary.get("missing_audio_count", 0))
        group["total_candidates"] += int(summary.get("total_candidates", 0))


def evaluation_generated_sources(config: EngineConfig) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    groups = _empty_source_groups()
    genre_groups: dict[str, dict[str, Any]] = {}
    total_audio_ready = 0
    total_candidates = 0
    for ranking in sorted(list_rankings(config), key=lambda item: item.get("created_at") or "", reverse=True):
        ranking_path = Path(str(ranking.get("path", "")))
        ranking_payload = _safe_load_json(ranking_path)
        rows = []
        for candidate in sorted(ranking.get("candidates", []), key=lambda item: int(item.get("rank") or 999)):
            wav_url = candidate.get("wav_download_url")
            mp3_url = candidate.get("mp3_download_url")
            audio_ready = bool(wav_url or mp3_url)
            total_candidates += 1
            if audio_ready:
                total_audio_ready += 1
            raw_candidate = next(
                (
                    item
                    for item in ranking_payload.get("candidates", [])
                    if item.get("candidate_id") == candidate.get("candidate_id")
                ),
                {},
            )
            rows.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "rank": candidate.get("rank"),
                    "score": candidate.get("score"),
                    "seed": candidate.get("seed"),
                    "genre": _candidate_genre(ranking_payload, raw_candidate),
                    "audio_ready": audio_ready,
                    "needs_render": not audio_ready and bool(candidate.get("midi_path")),
                    "midi_path": candidate.get("midi_path"),
                    "midi_download_url": candidate.get("midi_download_url"),
                    "wav_download_url": wav_url,
                    "mp3_download_url": mp3_url,
                    "duration_seconds": candidate.get("duration_seconds"),
                    "note_count": candidate.get("note_count"),
                }
            )
        audio_ready_count = sum(1 for item in rows if item["audio_ready"])
        source = {
                "source_type": "ranking",
                "source_id": ranking.get("ranked_id"),
                "label": ranking.get("label"),
                "created_at": ranking.get("created_at"),
                "generation_mode": ranking.get("generation_mode"),
                "generation_mode_label": ranking.get("generation_mode_label"),
                "path": ranking.get("path"),
                "total_candidates": len(rows),
                "audio_ready_count": audio_ready_count,
                "missing_audio_count": len(rows) - audio_ready_count,
                "max_selectable": audio_ready_count,
                "candidates": rows,
        }
        source["source_group"] = _source_group(source)
        source["genres"] = _source_genre_summary(source)
        sources.append(source)
        group = groups[source["source_group"]]
        group["sources"].append(source)
        group["total_audio_ready"] += audio_ready_count
        group["total_missing_audio"] += len(rows) - audio_ready_count
        _add_source_to_genre_groups(genre_groups, source)
    render_rows = []
    for render in sorted(list_renders(config), key=lambda item: item.get("created_at") or "", reverse=True):
        if render.get("source_ranked_id") or render.get("source_candidate_id"):
            continue
        if render.get("wav_download_url") or render.get("mp3_download_url"):
            render_rows.append(render)
    if render_rows:
        total_audio_ready += len(render_rows)
        total_candidates += len(render_rows)
        source = {
                "source_type": "renders",
                "source_id": "standalone-renders",
                "label": "Renders sueltos",
                "created_at": render_rows[0].get("created_at"),
                "generation_mode": "render",
                "generation_mode_label": "Render de audio",
                "source_group": "renders",
                "total_candidates": len(render_rows),
                "audio_ready_count": len(render_rows),
                "missing_audio_count": 0,
                "max_selectable": len(render_rows),
                "candidates": [
                    {
                        "candidate_id": render.get("label"),
                        "rank": index,
                        "score": None,
                        "seed": None,
                        "genre": "unknown",
                        "audio_ready": True,
                        "needs_render": False,
                        "midi_path": render.get("source_midi"),
                        "wav_download_url": render.get("wav_download_url"),
                        "mp3_download_url": render.get("mp3_download_url"),
                        "duration_seconds": None,
                        "note_count": None,
                    }
                    for index, render in enumerate(render_rows, start=1)
                ],
        }
        source["genres"] = _source_genre_summary(source)
        sources.append(source)
        groups["renders"]["sources"].append(source)
        groups["renders"]["total_audio_ready"] += len(render_rows)
        _add_source_to_genre_groups(genre_groups, source)
    return {
        "sources": sources,
        "groups": groups,
        "genre_groups": dict(sorted(genre_groups.items())),
        "total_sources": len(sources),
        "total_candidates": total_candidates,
        "total_audio_ready": total_audio_ready,
    }


def create_evaluation_from_results(
    config: EngineConfig,
    *,
    selections: list[dict[str, Any]],
    genre_selections: dict[str, list[dict[str, Any]]] | None = None,
    target_per_genre: int = 20,
    pairing_strategy: str = "same_genre_round_robin",
    real_audio_root: Path | None = None,
    output_name: str = "generated_results",
) -> dict[str, Any]:
    matrix_selections: list[dict[str, Any]] = []
    for genre, rows in (genre_selections or {}).items():
        for row in rows or []:
            limit = int(row.get("limit") or row.get("count") or 0)
            if limit <= 0:
                continue
            matrix_selections.append(
                {
                    "source_type": row.get("source_type") or "ranking",
                    "source_id": row.get("source_id"),
                    "limit": limit,
                    "candidate_ids": row.get("candidate_ids") or [],
                    "genre": genre,
                }
            )
    selections = matrix_selections or selections
    if not selections:
        raise RuntimeError("Selecciona al menos una corrida con audio renderizado.")
    if pairing_strategy != "same_genre_round_robin":
        raise RuntimeError("La estrategia soportada actualmente es same_genre_round_robin.")
    real_root = Path(real_audio_root).expanduser().resolve() if real_audio_root else _default_real_root(config)
    if not real_root.exists():
        raise RuntimeError(f"Carpeta de audio real no encontrada: {real_root}")

    evaluation_id = create_id(output_name, prefix="evaluation")
    evaluation_dir = config.data_dir / "evaluations" / evaluation_id
    generated_root = evaluation_dir / "generated"
    copied_real_root = evaluation_dir / "real"
    metrics_root = evaluation_dir / "metrics"
    plots_root = evaluation_dir / "plots"
    for folder in [generated_root, copied_real_root, metrics_root, plots_root]:
        folder.mkdir(parents=True, exist_ok=True)

    ranking_index = {item.get("ranked_id"): item for item in list_rankings(config)}
    standalone_renders = [
        item
        for item in sorted(list_renders(config), key=lambda row: row.get("created_at") or "", reverse=True)
        if (item.get("wav_path") or item.get("mp3_path"))
        and not item.get("source_ranked_id")
        and not item.get("source_candidate_id")
    ]
    selected_tracks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    genre_counts: dict[str, int] = {}
    selected_by_genre: dict[str, int] = {}

    for selection in selections:
        source_id = str(selection.get("source_id") or "")
        limit = max(0, int(selection.get("limit") or 0))
        requested_genre = str(selection.get("genre") or "")
        candidate_ids = {str(value) for value in selection.get("candidate_ids") or [] if value}
        if not source_id or limit <= 0:
            continue
        if str(selection.get("source_type") or "") == "renders" or source_id == "standalone-renders":
            for index, render in enumerate(standalone_renders[:limit], start=1):
                wav_path = Path(str(render.get("wav_path", ""))) if render.get("wav_path") else None
                mp3_path = Path(str(render.get("mp3_path", ""))) if render.get("mp3_path") else None
                if (not wav_path or not wav_path.is_file()) and (not mp3_path or not mp3_path.is_file()):
                    continue
                genre = "unknown"
                if requested_genre and requested_genre != genre:
                    continue
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                selected_by_genre[genre] = selected_by_genre.get(genre, 0) + 1
                track_id = f"render_{index:03d}_{str(render.get('label') or 'audio').replace('/', '_')}"
                candidate_dir = generated_root / genre / track_id
                midi_path = Path(str(render.get("source_midi", ""))) if render.get("source_midi") else None
                paths = {
                    "midi_path": _copy_file(midi_path, candidate_dir / f"{track_id}.mid") if midi_path else None,
                    "tokens_path": None,
                    "wav_path": _copy_file(wav_path, candidate_dir / f"{track_id}.wav") if wav_path and wav_path.is_file() else None,
                    "mp3_path": _copy_file(mp3_path, candidate_dir / f"{track_id}.mp3") if mp3_path and mp3_path.is_file() else None,
                }
                metadata = {
                    "track_id": track_id,
                    "genre": genre,
                    "prompt": "generated music",
                    "status": "completed",
                    "source": {
                        "source_type": "renders",
                        "source_id": source_id,
                        "candidate_id": render.get("label"),
                        "generation_mode": render.get("generation_mode"),
                        "generation_mode_label": render.get("generation_mode_label"),
                    },
                    "candidate_id": render.get("label"),
                    "rank": index,
                    "score": None,
                    "seed": None,
                    "metrics": {},
                    "paths": paths,
                }
                _json_dump(candidate_dir / "metadata.json", metadata)
                selected_tracks.append(metadata)
            continue
        ranking = ranking_index.get(source_id)
        if not ranking:
            errors.append({"source_id": source_id, "error": "Corrida no encontrada."})
            continue
        ranking_payload = _safe_load_json(Path(str(ranking.get("path", ""))))
        raw_candidates = {
            item.get("candidate_id"): item
            for item in ranking_payload.get("candidates", [])
            if item.get("candidate_id")
        }
        candidates = sorted(ranking.get("candidates", []), key=lambda item: int(item.get("rank") or 999))
        if candidate_ids:
            candidates = [item for item in candidates if str(item.get("candidate_id")) in candidate_ids]
        selected = []
        for candidate in candidates:
            if len(selected) >= limit:
                break
            raw_candidate = raw_candidates.get(candidate.get("candidate_id"), {})
            genre = _candidate_genre(ranking_payload, raw_candidate)
            if requested_genre and genre != requested_genre:
                continue
            wav_path, mp3_path = _candidate_audio_paths(raw_candidate)
            if not wav_path and not mp3_path:
                errors.append(
                    {
                        "source_id": source_id,
                        "candidate_id": candidate.get("candidate_id"),
                        "error": "La candidata no tiene WAV/MP3 renderizado.",
                    }
                )
                continue
            selected.append((candidate, raw_candidate, wav_path, mp3_path))
        for candidate, raw_candidate, wav_path, mp3_path in selected:
            genre = _candidate_genre(ranking_payload, raw_candidate)
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            selected_by_genre[genre] = selected_by_genre.get(genre, 0) + 1
            track_id = f"{source_id}_{candidate.get('candidate_id')}".replace("/", "_")
            candidate_dir = generated_root / genre / track_id
            midi_path = Path(str(raw_candidate.get("midi_path", candidate.get("midi_path", ""))))
            tokens_path = Path(str(raw_candidate.get("tokens_path", ""))) if raw_candidate.get("tokens_path") else None
            paths = {
                "midi_path": _copy_file(midi_path, candidate_dir / f"{track_id}.mid"),
                "tokens_path": _copy_file(tokens_path, candidate_dir / f"{track_id}.tokens.json") if tokens_path else None,
                "wav_path": _copy_file(wav_path, candidate_dir / f"{track_id}.wav") if wav_path else None,
                "mp3_path": _copy_file(mp3_path, candidate_dir / f"{track_id}.mp3") if mp3_path else None,
            }
            metadata = {
                "track_id": track_id,
                "genre": genre,
                "prompt": raw_candidate.get("generation", {}).get("condition_genre") or genre,
                "status": "completed",
                "source": {
                    "source_type": "ranking",
                    "source_id": source_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "generation_mode": ranking.get("generation_mode"),
                    "generation_mode_label": ranking.get("generation_mode_label"),
                },
                "candidate_id": candidate.get("candidate_id"),
                "rank": candidate.get("rank"),
                "score": candidate.get("score"),
                "seed": candidate.get("seed"),
                "metrics": raw_candidate.get("metrics", {}),
                "paths": paths,
            }
            _json_dump(candidate_dir / "metadata.json", metadata)
            selected_tracks.append(metadata)

    if not selected_tracks:
        raise RuntimeError("No hay canciones evaluables. Renderiza WAV/MP3 antes de calcular métricas.")

    copied_real_counts = _copy_reference_audio(real_root, copied_real_root, genre_counts)
    pairs = _make_same_genre_pairs(config, selected_tracks=selected_tracks, copied_real_root=copied_real_root)
    _json_dump(metrics_root / "pair_metrics.json", {"pairs": pairs})
    manifest = {
        "schema_version": "music-evaluation-manifest-v1",
        "evaluation_id": evaluation_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed" if not errors else "completed_with_warnings",
        "source": "generated_results",
        "selection_mode": "genre_run_matrix" if matrix_selections else "source_limit",
        "pairing_strategy": pairing_strategy,
        "target_per_genre": target_per_genre,
        "real_audio_root": str(real_root),
        "distribution": genre_counts,
        "reference_distribution": copied_real_counts,
        "selected_sources": selections,
        "selected_by_genre": selected_by_genre,
        "paths": {
            "root": str(evaluation_dir),
            "generated": str(generated_root),
            "real": str(copied_real_root),
            "metrics": str(metrics_root),
            "plots": str(plots_root),
        },
        "tracks": selected_tracks,
        "pairs": pairs,
        "errors": errors,
        "valid_tracks": len(selected_tracks),
    }
    _json_dump(evaluation_dir / "manifest.json", manifest)
    return manifest


def generate_evaluation_batch(
    config: EngineConfig,
    *,
    model_path: Path,
    distribution: dict[str, int] | None = None,
    real_audio_root: Path | None = None,
    duration_seconds: float = 30.0,
    output_name: str = "evaluation_batch",
    seed: int | None = 42,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    export_layers: bool = True,
    render_audio: bool = True,
    render_engine: str = "auto",
    export_mp3: bool = True,
    target_total: int = 100,
) -> dict[str, Any]:
    source_model = Path(model_path).expanduser().resolve()
    if not source_model.exists():
        raise RuntimeError(f"Modelo no encontrado: {source_model}")
    real_root = Path(real_audio_root).expanduser().resolve() if real_audio_root else _default_real_root(config)
    normalized_distribution = _normalize_distribution(
        config,
        distribution,
        real_audio_root=real_root,
        target_total=target_total,
    )
    evaluation_id = create_id(output_name, prefix="evaluation")
    evaluation_dir = config.data_dir / "evaluations" / evaluation_id
    generated_root = evaluation_dir / "generated"
    copied_real_root = evaluation_dir / "real"
    metrics_root = evaluation_dir / "metrics"
    plots_root = evaluation_dir / "plots"
    for folder in [generated_root, copied_real_root, metrics_root, plots_root]:
        folder.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "music-evaluation-manifest-v1",
        "evaluation_id": evaluation_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "model_path": str(source_model),
        "real_audio_root": str(real_root),
        "distribution": normalized_distribution,
        "duration_seconds": duration_seconds,
        "sampling": {
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        "render": {
            "render_audio": render_audio,
            "render_engine": render_engine,
            "export_mp3": export_mp3,
        },
        "paths": {
            "root": str(evaluation_dir),
            "generated": str(generated_root),
            "real": str(copied_real_root),
            "metrics": str(metrics_root),
            "plots": str(plots_root),
        },
        "tracks": [],
        "errors": [],
    }
    manifest_path = evaluation_dir / "manifest.json"

    # Copy selected real reference files into the formal evaluation folder.
    for genre, count in normalized_distribution.items():
        real_files = _audio_files(real_root / genre)[:count]
        for index, real_file in enumerate(real_files, start=1):
            _copy_file(real_file, copied_real_root / genre / f"real_{index:03d}{real_file.suffix.lower()}")

    total = sum(normalized_distribution.values())
    current = 0
    for genre, count in normalized_distribution.items():
        for local_index in range(1, count + 1):
            current += 1
            track_id = f"{genre}_{local_index:03d}"
            candidate_dir = generated_root / genre / track_id
            metadata_path = candidate_dir / "metadata.json"
            track_seed = (seed + current - 1) if seed is not None else None
            metadata: dict[str, Any] = {
                "track_id": track_id,
                "genre": genre,
                "index": local_index,
                "seed": track_seed,
                "prompt": f"{genre} music",
                "status": "running",
            }
            try:
                ranking = generate_ranked_candidates(
                    config,
                    model_path=source_model,
                    duration_seconds=duration_seconds,
                    output_name=f"{output_name}_{track_id}",
                    candidates=1,
                    seed=track_seed,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    condition_genre=genre,
                    feature_tokens=[f"genre:{genre}", "evaluation:batch"],
                    embedding_path=None,
                    export_layers=export_layers,
                    render_best=render_audio,
                    render_engine=render_engine,
                    export_mp3=export_mp3,
                )
                candidate = (ranking.get("candidates") or [{}])[0]
                midi_path = Path(str(candidate.get("midi_path", "")))
                tokens_path = Path(str(candidate.get("tokens_path", "")))
                wav_path, mp3_path = _candidate_audio_from_render(candidate.get("render"))
                saved = {
                    "midi_path": _copy_file(midi_path, candidate_dir / f"{track_id}.mid"),
                    "tokens_path": _copy_file(tokens_path, candidate_dir / f"{track_id}.tokens.json"),
                    "wav_path": _copy_file(wav_path, candidate_dir / f"{track_id}.wav") if wav_path else None,
                    "mp3_path": _copy_file(mp3_path, candidate_dir / f"{track_id}.mp3") if mp3_path else None,
                    "ranking_path": str(ranking.get("path")),
                }
                layers_dir = candidate_dir / "layers"
                layer_midis = {}
                for layer, layer_value in (candidate.get("layer_midis") or {}).items():
                    layer_path = Path(str(layer_value))
                    copied = _copy_file(layer_path, layers_dir / f"{layer}.mid")
                    if copied:
                        layer_midis[layer] = copied
                metadata.update(
                    {
                        "status": "completed",
                        "candidate_id": candidate.get("candidate_id"),
                        "score": candidate.get("score"),
                        "metrics": candidate.get("metrics", {}),
                        "paths": saved,
                        "layer_midis": layer_midis,
                    }
                )
            except Exception as exc:  # keep batch going per song
                metadata.update({"status": "failed", "error": str(exc)})
                manifest["errors"].append({"track_id": track_id, "genre": genre, "error": str(exc)})
            candidate_dir.mkdir(parents=True, exist_ok=True)
            _json_dump(metadata_path, metadata)
            manifest["tracks"].append(metadata)
            manifest["progress"] = {"completed": current, "total": total}
            _json_dump(manifest_path, manifest)

    manifest["status"] = "completed" if not manifest["errors"] else "completed_with_errors"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["valid_tracks"] = len([row for row in manifest["tracks"] if row.get("status") == "completed"])
    completed_tracks = [row for row in manifest["tracks"] if row.get("status") == "completed"]
    pairs = _make_same_genre_pairs(config, selected_tracks=completed_tracks, copied_real_root=copied_real_root)
    manifest["pairs"] = pairs
    _json_dump(metrics_root / "pair_metrics.json", {"pairs": pairs})
    _json_dump(manifest_path, manifest)
    return manifest


def _latest_classifier(config: EngineConfig) -> Path | None:
    candidates = sorted((config.data_dir / "models" / "audio_classifier").glob("*/classifier.json"))
    return candidates[-1] if candidates else None


def _save_probabilities(
    classifier: AudioCentroidClassifier,
    files: list[Path],
    npy_path: Path,
    json_path: Path,
    csv_path: Path,
) -> np.ndarray:
    probs = classifier.predict_batch(files)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, probs)
    rows = [
        {"path": str(path), "probabilities": probs[index].astype(float).tolist()}
        for index, path in enumerate(files)
    ]
    json_path.write_text(json.dumps({"labels": classifier.labels, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", *classifier.labels]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(files):
            writer.writerow({"path": str(path), **{label: float(probs[index, i]) for i, label in enumerate(classifier.labels)}})
    return probs


def _distribution_from_probs(probs: np.ndarray) -> list[float]:
    if probs.size == 0:
        return []
    dist = probs.mean(axis=0)
    total = dist.sum()
    if total <= 0:
        return []
    return (dist / total).astype(float).tolist()


def _compute_kld(real_probs: np.ndarray, generated_probs: np.ndarray) -> dict[str, Any]:
    from hybrid_music_engine.new_metrics.kld_metric import (
        calcular_kld_desde_distribuciones,
        promediar_predicciones,
    )
    p = promediar_predicciones(real_probs).astype(float)
    q = promediar_predicciones(generated_probs).astype(float)
    return {
        "kld": calcular_kld_desde_distribuciones(p, q),
        "real_distribution": p.tolist(),
        "generated_distribution": q.tolist(),
    }


def _compute_pair_kld(real_probabilities: list[float], generated_probabilities: list[float]) -> float | None:
    if not real_probabilities or not generated_probabilities:
        return None
    real = np.asarray(real_probabilities, dtype=np.float64).reshape(1, -1)
    generated = np.asarray(generated_probabilities, dtype=np.float64).reshape(1, -1)
    return float(_compute_kld(real, generated)["kld"])


def _extract_mel_summary(audio_path: Path, *, sample_rate: int = 22050, max_seconds: float | None = 45.0) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa es necesario para FAD mel.") from exc
    load_duration = float(max_seconds) if max_seconds and max_seconds > 0 else 45.0
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=load_duration)
    if y.size < 256:
        raise RuntimeError(f"Audio demasiado corto para FAD: {audio_path}")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    db = librosa.power_to_db(mel)
    return np.concatenate([db.mean(axis=1), db.std(axis=1)]).astype(np.float64)


def _frechet_from_features(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    try:
        from scipy import linalg
    except ImportError:
        linalg = None
    mu_r = real_features.mean(axis=0)
    mu_g = generated_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False) if real_features.shape[0] > 1 else np.eye(real_features.shape[1]) * 1e-6
    sigma_g = np.cov(generated_features, rowvar=False) if generated_features.shape[0] > 1 else np.eye(generated_features.shape[1]) * 1e-6
    diff = mu_r - mu_g
    if linalg is None:
        return float(diff @ diff + np.trace(sigma_r + sigma_g))
    sqrt_prod, _ = linalg.sqrtm(sigma_r @ sigma_g + np.eye(sigma_r.shape[0]) * 1e-6, disp=False)
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * sqrt_prod))


def _compute_fad_mel(real_files: list[Path], generated_files: list[Path], output_path: Path) -> dict[str, Any]:
    if not real_files or not generated_files:
        raise RuntimeError("FAD requiere audio real y generado.")
    real_features = np.vstack([_extract_mel_summary(path) for path in real_files])
    generated_features = np.vstack([_extract_mel_summary(path) for path in generated_files])
    payload = {
        "extractor": "mel",
        "fad": _frechet_from_features(real_features, generated_features),
        "real_files": len(real_files),
        "generated_files": len(generated_files),
    }
    _json_dump(output_path, payload)
    return payload


def _compute_pair_fad(real_audio: Path, generated_audio: Path, *, real_max_seconds: float | None = None) -> float:
    real_features = _extract_mel_summary(real_audio, max_seconds=real_max_seconds).reshape(1, -1)
    generated_features = _extract_mel_summary(generated_audio).reshape(1, -1)
    return _frechet_from_features(real_features, generated_features)


def _compute_kad_mel(real_files: list[Path], generated_files: list[Path], output_path: Path) -> dict[str, Any]:
    """
    KAD (Kernel Audio Distance, MMD-RBF) sobre los mismos resúmenes mel que usa
    FAD en la ruta integrada. Reutiliza la implementación de new_metrics.kad_score
    para mantener una sola definición de la métrica.
    """
    from hybrid_music_engine.new_metrics.kad_score import compute_kad_from_embeddings

    if len(real_files) < 2 or len(generated_files) < 2:
        raise RuntimeError("KAD requiere al menos 2 clips reales y 2 generados.")
    real_features = np.vstack([_extract_mel_summary(path) for path in real_files])
    generated_features = np.vstack([_extract_mel_summary(path) for path in generated_files])
    core = compute_kad_from_embeddings(real_features, generated_features)
    payload = {
        "extractor": "mel",
        "kad": core["kad"],
        "sigma": core["sigma"],
        "real_files": len(real_files),
        "generated_files": len(generated_files),
    }
    _json_dump(output_path, payload)
    return payload


def _tempo_for_files(files: list[Path]) -> dict[str, Any]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa es necesario para métricas de tempo.") from exc
    rows = []
    for path in files:
        try:
            y, sr = librosa.load(str(path), sr=22050, mono=True, duration=45.0)
            tempo = float(np.asarray(librosa.beat.tempo(y=y, sr=sr)).ravel()[0]) if y.size else 0.0
            rows.append({"path": str(path), "tempo": tempo})
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc), "tempo": None})
    valid = [float(row["tempo"]) for row in rows if row.get("tempo") is not None]
    return {
        "rows": rows,
        "mean": float(np.mean(valid)) if valid else None,
        "std": float(np.std(valid)) if valid else None,
        "count": len(valid),
    }


def _audio_descriptor(audio_path: Path, *, max_seconds: float | None = 45.0) -> dict[str, Any]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa es necesario para comparar canciones.") from exc
    load_duration = float(max_seconds) if max_seconds and max_seconds > 0 else 45.0
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=load_duration)
    if y.size < 256:
        raise RuntimeError(f"Audio demasiado corto para comparar: {audio_path}")
    try:
        tempo = float(np.asarray(librosa.beat.tempo(y=y, sr=sr)).ravel()[0])
    except Exception:
        tempo = None
    return {
        "duration_seconds": float(librosa.get_duration(y=y, sr=sr)),
        "tempo": tempo,
        "rms_mean": float(np.mean(librosa.feature.rms(y=y))) if y.size else None,
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(float(a @ b) / denom, -1.0, 1.0))


def _midi_metrics_for_files(files: list[Path]) -> dict[str, Any]:
    rows = []
    for path in files:
        try:
            metrics = analyze_midi_quality(path)
            rows.append({"path": str(path), **metrics})
        except RuntimeError as exc:
            rows.append({"path": str(path), "error": str(exc)})
    valid = [row for row in rows if "error" not in row]
    return {
        "rows": rows,
        "count": len(valid),
        "valid_midi_rate": (sum(1 for row in valid if row.get("valid_midi")) / len(valid)) if valid else 0.0,
        "mean_quality_score": float(np.mean([float(row.get("quality_score", 0.0)) for row in valid])) if valid else 0.0,
        "mean_note_count": float(np.mean([float(row.get("note_count", 0.0)) for row in valid])) if valid else 0.0,
        "mean_duration_seconds": float(np.mean([float(row.get("duration_seconds", 0.0)) for row in valid])) if valid else 0.0,
        "mean_note_density": float(np.mean([float(row.get("note_density_per_second", 0.0)) for row in valid])) if valid else 0.0,
        "mean_pitch_diversity": float(np.mean([float(row.get("pitch_class_diversity", 0.0)) for row in valid])) if valid else 0.0,
        "mean_rhythm_diversity": float(np.mean([float(row.get("rhythm_diversity", 0.0)) for row in valid])) if valid else 0.0,
    }


def _compute_pair_and_genre_metrics(
    *,
    config: EngineConfig,
    evaluation_dir: Path,
    classifier: AudioCentroidClassifier | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _safe_load_json(evaluation_dir / "manifest.json")
    pairs = manifest.get("pairs") or (_safe_load_json(evaluation_dir / "metrics" / "pair_metrics.json").get("pairs") or [])
    if not pairs:
        pairs = _ensure_evaluation_pairs(config, evaluation_dir)
    rows: list[dict[str, Any]] = []
    genre_rows: dict[str, list[dict[str, Any]]] = {}
    labels = classifier.labels if classifier else []
    for pair in pairs:
        genre = str(pair.get("genre") or "unknown")
        generated = pair.get("generated") or {}
        original = pair.get("original") or {}
        gen_audio = Path(str(generated.get("audio_path", ""))) if generated.get("audio_path") else None
        real_audio = Path(str(original.get("audio_path", ""))) if original and original.get("audio_path") else None
        midi_path = Path(str(generated.get("midi_path", ""))) if generated.get("midi_path") else None
        row: dict[str, Any] = {
            "pair_id": pair.get("pair_id"),
            "genre": genre,
            "track_id": pair.get("track_id"),
            "candidate_id": pair.get("candidate_id"),
            "source": pair.get("source") or {},
            "rank": pair.get("rank"),
            "score": pair.get("score"),
            "generated": generated,
            "original": original,
            "metrics": {},
            "errors": [],
        }
        try:
            if not gen_audio or not gen_audio.exists():
                raise RuntimeError("Falta audio generado.")
            if not real_audio or not real_audio.exists():
                raise RuntimeError("Falta audio original.")
            gen_desc = _audio_descriptor(gen_audio)
            # El audio real se mide a la MISMA duracion que el generado para que la
            # comparacion real vs generado sea directa y justa por genero.
            comparison_seconds = gen_desc.get("duration_seconds")
            real_desc = _audio_descriptor(real_audio, max_seconds=comparison_seconds)
            similarity = _cosine_similarity(
                _extract_mel_summary(real_audio, max_seconds=comparison_seconds),
                _extract_mel_summary(gen_audio),
            )
            pair_fad = _compute_pair_fad(real_audio, gen_audio, real_max_seconds=comparison_seconds)
            row["metrics"].update(
                {
                    "audio_similarity": similarity,
                    "fad": pair_fad,
                    "comparison_seconds": comparison_seconds,
                    "generated_duration_seconds": gen_desc.get("duration_seconds"),
                    "duration_seconds": gen_desc.get("duration_seconds"),
                    "original_duration_seconds": real_desc.get("duration_seconds"),
                    "duration_delta_seconds": (
                        float(gen_desc["duration_seconds"]) - float(real_desc["duration_seconds"])
                        if gen_desc.get("duration_seconds") is not None and real_desc.get("duration_seconds") is not None
                        else None
                    ),
                    "generated_tempo": gen_desc.get("tempo"),
                    "tempo": gen_desc.get("tempo"),
                    "original_tempo": real_desc.get("tempo"),
                    "tempo_delta": (
                        float(gen_desc["tempo"]) - float(real_desc["tempo"])
                        if gen_desc.get("tempo") is not None and real_desc.get("tempo") is not None
                        else None
                    ),
                    "generated_rms_mean": gen_desc.get("rms_mean"),
                    "original_rms_mean": real_desc.get("rms_mean"),
                }
            )
            if midi_path and midi_path.exists():
                try:
                    midi_metrics = analyze_midi_quality(midi_path)
                    row["metrics"]["midi"] = midi_metrics
                    row["metrics"].update(
                        {
                            "midi_quality_score": midi_metrics.get("quality_score"),
                            "valid_midi": midi_metrics.get("valid_midi"),
                            "valid_midi_score": 1.0 if midi_metrics.get("valid_midi") else 0.0,
                            "note_count": midi_metrics.get("note_count"),
                            "note_density_per_second": midi_metrics.get("note_density_per_second"),
                            "pitch_class_diversity": midi_metrics.get("pitch_class_diversity"),
                            "pitch_diversity": midi_metrics.get("pitch_class_diversity"),
                            "rhythm_diversity": midi_metrics.get("rhythm_diversity"),
                        }
                    )
                except RuntimeError as exc:
                    row["errors"].append({"metric": "midi", "error": str(exc)})
            else:
                row["errors"].append({"metric": "midi", "error": "Falta MIDI para calcular métricas musicales."})
            if classifier:
                try:
                    gen_probs = classifier.predict_proba(gen_audio)
                    real_probs = classifier.predict_proba(real_audio)
                    target_index = labels.index(genre) if genre in labels else None
                    generated_target_probability = gen_probs[target_index] if target_index is not None else None
                    original_target_probability = real_probs[target_index] if target_index is not None else None
                    row["metrics"]["classifier"] = {
                        "labels": labels,
                        "generated_probabilities": gen_probs,
                        "original_probabilities": real_probs,
                        "generated_target_probability": generated_target_probability,
                        "original_target_probability": original_target_probability,
                    }
                    pair_kld = _compute_pair_kld(real_probs, gen_probs)
                    row["metrics"]["kld"] = pair_kld
                    row["metrics"]["generated_target_probability"] = generated_target_probability
                    row["metrics"]["original_target_probability"] = original_target_probability
                except RuntimeError as exc:
                    row["errors"].append({"metric": "classifier", "error": str(exc)})
            else:
                row["errors"].append({"metric": "classifier", "error": "Clasificador no disponible para probabilidad de género."})
            row["metrics"]["reward_score"] = row.get("score")
            quality_components = []
            if row["metrics"].get("midi_quality_score") is not None:
                quality_components.append(float(row["metrics"]["midi_quality_score"]))
            if row["metrics"].get("valid_midi_score") is not None:
                quality_components.append(float(row["metrics"]["valid_midi_score"]))
            if row.get("score") is not None:
                quality_components.append(max(0.0, min(1.0, float(row["score"]))))
            row["metrics"]["quality_general"] = float(np.mean(quality_components)) if quality_components else None
        except Exception as exc:
            row["errors"].append({"metric": "pair", "error": str(exc)})
        rows.append(row)
        genre_rows.setdefault(genre, []).append(row)

    def _mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def _collect(metric_rows: list[dict[str, Any]], key: str, *, absolute: bool = False) -> list[float]:
        out: list[float] = []
        for row in metric_rows:
            value = row.get(key)
            if value is None:
                continue
            number = float(value)
            out.append(abs(number) if absolute else number)
        return out

    genre_summary: dict[str, Any] = {}
    genre_comparison: dict[str, Any] = {}
    for genre, genre_pair_rows in sorted(genre_rows.items()):
        metric_rows = [row.get("metrics") or {} for row in genre_pair_rows]
        similarities = [float(row["audio_similarity"]) for row in metric_rows if row.get("audio_similarity") is not None]
        fads = [float(row["fad"]) for row in metric_rows if row.get("fad") is not None]
        klds = [float(row["kld"]) for row in metric_rows if row.get("kld") is not None]
        qualities = [float(row["quality_general"]) for row in metric_rows if row.get("quality_general") is not None]
        valid_midi_scores = [float(row["valid_midi_score"]) for row in metric_rows if row.get("valid_midi_score") is not None]
        durations = [float(row["duration_seconds"]) for row in metric_rows if row.get("duration_seconds") is not None]
        tempos = [float(row["tempo"]) for row in metric_rows if row.get("tempo") is not None]
        tempo_deltas = [abs(float(row["tempo_delta"])) for row in metric_rows if row.get("tempo_delta") is not None]
        duration_deltas = [abs(float(row["duration_delta_seconds"])) for row in metric_rows if row.get("duration_delta_seconds") is not None]
        target_probs = [
            float(row["generated_target_probability"])
            for row in metric_rows
            if row.get("generated_target_probability") is not None
        ]
        reward_scores = [float(row["reward_score"]) for row in metric_rows if row.get("reward_score") is not None]
        densities = [float(row["note_density_per_second"]) for row in metric_rows if row.get("note_density_per_second") is not None]
        pitch_diversities = [float(row["pitch_class_diversity"]) for row in metric_rows if row.get("pitch_class_diversity") is not None]
        rhythm_diversities = [float(row["rhythm_diversity"]) for row in metric_rows if row.get("rhythm_diversity") is not None]
        midi_rows = [
            row.get("midi") or {}
            for row in metric_rows
            if isinstance(row.get("midi"), dict) and not row.get("midi", {}).get("error")
        ]
        genre_summary[genre] = {
            "genre": genre,
            "pairs": len(genre_pair_rows),
            "valid_pairs": sum(1 for row in genre_pair_rows if not row.get("errors")),
            "audio_similarity_mean": float(np.mean(similarities)) if similarities else None,
            "fad_mean": float(np.mean(fads)) if fads else None,
            "kld_mean": float(np.mean(klds)) if klds else None,
            "quality_general_mean": float(np.mean(qualities)) if qualities else None,
            "valid_midi_rate": float(np.mean(valid_midi_scores)) if valid_midi_scores else None,
            "duration_mean": float(np.mean(durations)) if durations else None,
            "tempo_mean": float(np.mean(tempos)) if tempos else None,
            "tempo_delta_mean": float(np.mean(tempo_deltas)) if tempo_deltas else None,
            "duration_delta_mean": float(np.mean(duration_deltas)) if duration_deltas else None,
            "midi_quality_mean": float(np.mean([float(row.get("quality_score", 0.0)) for row in midi_rows])) if midi_rows else None,
            "target_probability_mean": float(np.mean(target_probs)) if target_probs else None,
            "reward_mean": float(np.mean(reward_scores)) if reward_scores else None,
            "note_density_mean": float(np.mean(densities)) if densities else None,
            "pitch_diversity_mean": float(np.mean(pitch_diversities)) if pitch_diversities else None,
            "rhythm_diversity_mean": float(np.mean(rhythm_diversities)) if rhythm_diversities else None,
            "errors": sum(len(row.get("errors") or []) for row in genre_pair_rows),
        }

        # Comparacion directa REAL vs GENERADO por genero. El audio real se midio a la
        # misma duracion que el generado (ver comparison_seconds), de modo que tempo,
        # duracion, sonoridad y confianza de genero son comparables lado a lado.
        comparison_durations = _collect(metric_rows, "comparison_seconds")
        real_block = {
            "tempo_mean": _mean(_collect(metric_rows, "original_tempo")),
            "duration_mean": _mean(_collect(metric_rows, "original_duration_seconds")),
            "rms_mean": _mean(_collect(metric_rows, "original_rms_mean")),
            "target_probability_mean": _mean(_collect(metric_rows, "original_target_probability")),
        }
        generated_block = {
            "tempo_mean": _mean(_collect(metric_rows, "tempo")),
            "duration_mean": _mean(_collect(metric_rows, "duration_seconds")),
            "rms_mean": _mean(_collect(metric_rows, "generated_rms_mean")),
            "target_probability_mean": _mean(_collect(metric_rows, "generated_target_probability")),
            "midi_quality_mean": _mean(_collect(metric_rows, "midi_quality_score")),
            "note_density_mean": _mean(_collect(metric_rows, "note_density_per_second")),
            "pitch_diversity_mean": _mean(_collect(metric_rows, "pitch_class_diversity")),
            "rhythm_diversity_mean": _mean(_collect(metric_rows, "rhythm_diversity")),
        }
        genre_comparison[genre] = {
            "genre": genre,
            "pairs": len(genre_pair_rows),
            "comparison_seconds_mean": _mean(comparison_durations),
            "real": real_block,
            "generated": generated_block,
            "distances": {
                "fad_mean": _mean(fads),
                "kld_mean": _mean(klds),
                "audio_similarity_mean": _mean(similarities),
                "tempo_delta_mean": _mean(_collect(metric_rows, "tempo_delta", absolute=True)),
                "duration_delta_mean": _mean(_collect(metric_rows, "duration_delta_seconds", absolute=True)),
            },
        }

    pair_payload = {
        "schema_version": "pair-metrics-v1",
        "evaluation_id": manifest.get("evaluation_id"),
        "pairing_strategy": manifest.get("pairing_strategy") or "same_genre_round_robin",
        "pairs": rows,
    }
    genre_payload = {
        "schema_version": "genre-summary-v2",
        "evaluation_id": manifest.get("evaluation_id"),
        "genres": genre_summary,
        "comparison": genre_comparison,
    }
    metrics_dir = evaluation_dir / "metrics"
    _json_dump(metrics_dir / "pair_metrics.json", pair_payload)
    _json_dump(metrics_dir / "genre_summary.json", genre_payload)
    return pair_payload, genre_payload


def _collect_prompt_pairs(generated_root: Path) -> list[dict[str, str]]:
    pairs = []
    for metadata_path in sorted(generated_root.glob("*/*/metadata.json")):
        metadata = _load_json(metadata_path)
        wav = ((metadata.get("paths") or {}).get("wav_path") or "")
        prompt = metadata.get("prompt") or metadata.get("genre") or "generated music"
        if wav and Path(str(wav)).exists():
            pairs.append({"audio": str(wav), "text": str(prompt)})
    return pairs


def run_evaluation_metrics(
    config: EngineConfig,
    *,
    evaluation_id: str | None = None,
    generated_root: Path | None = None,
    real_root: Path | None = None,
    prompts_path: Path | None = None,
    classifier_path: Path | None = None,
    train_classifier_if_missing: bool = True,
    metrics: list[str] | None = None,
    fad_extractor: str = "mel",
    clap_model: str = "clap",
    device: str = "cpu",
) -> dict[str, Any]:
    selected_metrics = metrics or ["fad", "kad", "kld", "tempo", "midi"]
    if evaluation_id:
        evaluation_dir = config.data_dir / "evaluations" / evaluation_id
        if not evaluation_dir.exists():
            raise RuntimeError(f"Evaluación no encontrada: {evaluation_id}")
        generated_root = generated_root or (evaluation_dir / "generated")
        real_root = real_root or (evaluation_dir / "real")
    else:
        evaluation_id = create_id("manual_evaluation", prefix="evaluation")
        evaluation_dir = config.data_dir / "evaluations" / evaluation_id
    generated_root = Path(generated_root).expanduser().resolve() if generated_root else None
    real_root = Path(real_root).expanduser().resolve() if real_root else _default_real_root(config)
    if not generated_root or not generated_root.exists():
        raise RuntimeError("Indica una carpeta generated_root válida o un evaluation_id existente.")
    if not real_root.exists():
        raise RuntimeError(f"Carpeta de audio real no encontrada: {real_root}")

    metrics_dir = evaluation_dir / "metrics"
    plots_dir = evaluation_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    real_audio = _audio_files(real_root)
    generated_audio = _audio_files(generated_root)
    generated_midis = _midi_files(generated_root)
    report: dict[str, Any] = {
        "schema_version": "music-evaluation-report-v1",
        "evaluation_id": evaluation_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "root": str(evaluation_dir),
            "generated": str(generated_root),
            "real": str(real_root),
            "metrics": str(metrics_dir),
            "plots": str(plots_dir),
        },
        "counts": {
            "real_audio": len(real_audio),
            "generated_audio": len(generated_audio),
            "generated_midis": len(generated_midis),
        },
        "metrics": {},
        "errors": [],
    }
    active_classifier: AudioCentroidClassifier | None = None

    if "fad" in selected_metrics:
        if fad_extractor != "mel":
            report["errors"].append({"metric": "fad", "error": "En API integrada el extractor operativo por defecto es mel; usa new_metrics CLI para vggish/clap."})
        try:
            global_fad = _compute_fad_mel(real_audio, generated_audio, metrics_dir / "fad.json")
            by_genre = {}
            for genre_dir in sorted(generated_root.iterdir() if generated_root.exists() else []):
                if not genre_dir.is_dir():
                    continue
                real_genre = real_root / genre_dir.name
                if real_genre.exists():
                    try:
                        by_genre[genre_dir.name] = _compute_fad_mel(
                            _audio_files(real_genre),
                            _audio_files(genre_dir),
                            metrics_dir / f"fad_{genre_dir.name}.json",
                        )
                    except Exception as exc:
                        by_genre[genre_dir.name] = {"error": str(exc)}
            global_fad["by_genre"] = by_genre
            _json_dump(metrics_dir / "fad.json", global_fad)
            report["metrics"]["fad"] = global_fad
        except Exception as exc:
            report["errors"].append({"metric": "fad", "error": str(exc)})
            report["metrics"]["fad"] = {"error": str(exc)}

    if "kad" in selected_metrics:
        try:
            global_kad = _compute_kad_mel(real_audio, generated_audio, metrics_dir / "kad.json")
            by_genre = {}
            for genre_dir in sorted(generated_root.iterdir() if generated_root.exists() else []):
                if not genre_dir.is_dir():
                    continue
                real_genre = real_root / genre_dir.name
                if real_genre.exists():
                    try:
                        by_genre[genre_dir.name] = _compute_kad_mel(
                            _audio_files(real_genre),
                            _audio_files(genre_dir),
                            metrics_dir / f"kad_{genre_dir.name}.json",
                        )
                    except Exception as exc:
                        by_genre[genre_dir.name] = {"error": str(exc)}
            global_kad["by_genre"] = by_genre
            _json_dump(metrics_dir / "kad.json", global_kad)
            report["metrics"]["kad"] = global_kad
        except Exception as exc:
            report["errors"].append({"metric": "kad", "error": str(exc)})
            report["metrics"]["kad"] = {"error": str(exc)}

    if "tempo" in selected_metrics or "tempos_std" in selected_metrics:
        try:
            tempo_payload = {
                "real": _tempo_for_files(real_audio),
                "generated": _tempo_for_files(generated_audio),
            }
            _json_dump(metrics_dir / "tempo.json", tempo_payload)
            report["metrics"]["tempo"] = tempo_payload
        except Exception as exc:
            report["errors"].append({"metric": "tempo", "error": str(exc)})

    if "midi" in selected_metrics:
        midi_payload = _midi_metrics_for_files(generated_midis)
        _json_dump(metrics_dir / "midi.json", midi_payload)
        report["metrics"]["midi"] = midi_payload

    if "kld" in selected_metrics:
        try:
            if classifier_path:
                selected_classifier = Path(classifier_path).expanduser().resolve()
            else:
                selected_classifier = _latest_classifier(config)
                if selected_classifier is None and train_classifier_if_missing:
                    trained = train_audio_classifier(config, real_audio_root=real_root, labels=sorted(_counts_by_genre(real_root)))
                    selected_classifier = Path(str(trained["path"]))
            if selected_classifier is None:
                raise RuntimeError("No hay clasificador entrenado para KLD.")
            classifier = AudioCentroidClassifier.load(selected_classifier)
            active_classifier = classifier
            real_probs = _save_probabilities(
                classifier,
                real_audio,
                metrics_dir / "probabilities_real.npy",
                metrics_dir / "probabilities_real.json",
                metrics_dir / "probabilities_real.csv",
            )
            generated_probs = _save_probabilities(
                classifier,
                generated_audio,
                metrics_dir / "probabilities_generated.npy",
                metrics_dir / "probabilities_generated.json",
                metrics_dir / "probabilities_generated.csv",
            )
            kld_payload = _compute_kld(real_probs, generated_probs)
            kld_payload["labels"] = classifier.labels
            kld_payload["classifier_path"] = str(selected_classifier)
            by_genre = {}
            for genre_dir in sorted(generated_root.iterdir() if generated_root.exists() else []):
                if not genre_dir.is_dir() or not (real_root / genre_dir.name).exists():
                    continue
                rg = _audio_files(real_root / genre_dir.name)
                gg = _audio_files(genre_dir)
                if rg and gg:
                    rp = classifier.predict_batch(rg)
                    gp = classifier.predict_batch(gg)
                    by_genre[genre_dir.name] = _compute_kld(rp, gp)
            kld_payload["by_genre"] = by_genre
            _json_dump(metrics_dir / "kld.json", kld_payload)
            report["metrics"]["kld"] = kld_payload
        except Exception as exc:
            report["errors"].append({"metric": "kld", "error": str(exc)})
            report["metrics"]["kld"] = {"error": str(exc)}

    if "clap" in selected_metrics:
        try:
            pairs = _collect_prompt_pairs(generated_root)
            if not pairs:
                raise RuntimeError("CLAP requiere audios generados con prompt asociado.")
            pairs_path = metrics_dir / "clap_pairs.json"
            pairs_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
            from hybrid_music_engine.new_metrics.clap_score import compute_clap_score
            clap_pairs = [(Path(item["audio"]), item["text"]) for item in pairs]
            clap_payload = compute_clap_score(
                clap_pairs,
                model_variant=clap_model,
                device=device,
                output_dir=metrics_dir / "clap_raw",
            )
            _json_dump(metrics_dir / "clap.json", clap_payload)
            report["metrics"]["clap"] = clap_payload
        except BaseException as exc:  # CLAP helper may call sys.exit on missing dependency.
            report["errors"].append({"metric": "clap", "error": str(exc)})
            report["metrics"]["clap"] = {"error": str(exc), "enabled": True}

    # Reward averages from metadata/ranking if present.
    scores = []
    for metadata_path in sorted(generated_root.glob("*/*/metadata.json")):
        try:
            metadata = _load_json(metadata_path)
            if metadata.get("score") is not None:
                scores.append(float(metadata["score"]))
        except Exception:
            continue
    reward_payload = {
        "mean_score": float(np.mean(scores)) if scores else None,
        "count": len(scores),
        "note": "Promedio de score/ranking disponible en metadata de candidatos; no sustituye FAD/KLD/CLAP.",
    }
    _json_dump(metrics_dir / "reward.json", reward_payload)
    report["metrics"]["reward"] = reward_payload

    if active_classifier is None:
        try:
            latest_classifier = _latest_classifier(config)
            active_classifier = AudioCentroidClassifier.load(latest_classifier) if latest_classifier else None
        except (OSError, RuntimeError, json.JSONDecodeError):
            active_classifier = None
    try:
        pair_payload, genre_payload = _compute_pair_and_genre_metrics(
            config=config,
            evaluation_dir=evaluation_dir,
            classifier=active_classifier,
        )
        report["metrics"]["pairs"] = {
            "count": len(pair_payload.get("pairs") or []),
            "path": str(metrics_dir / "pair_metrics.json"),
            "rows": pair_payload.get("pairs") or [],
        }
        report["metrics"]["genre_summary"] = genre_payload
    except Exception as exc:
        report["errors"].append({"metric": "pair_metrics", "error": str(exc)})

    clap_metrics = report["metrics"].get("clap") or {}
    summary = {
        "evaluation_id": evaluation_id,
        "fad": (report["metrics"].get("fad") or {}).get("fad"),
        "kad": (report["metrics"].get("kad") or {}).get("kad"),
        "kld": (report["metrics"].get("kld") or {}).get("kld"),
        "clap": clap_metrics.get("clap_score") or clap_metrics.get("mean_score") or clap_metrics.get("clap_score_mean"),
        "clap_std": clap_metrics.get("clap_std"),
        "pct_clap_above_025": clap_metrics.get("pct_above_threshold"),
        "reward_mean": reward_payload["mean_score"],
        "valid_midi_rate": (report["metrics"].get("midi") or {}).get("valid_midi_rate"),
        "quality_mean": (report["metrics"].get("midi") or {}).get("mean_quality_score"),
        "duration_mean": (report["metrics"].get("midi") or {}).get("mean_duration_seconds"),
        "note_density_mean": (report["metrics"].get("midi") or {}).get("mean_note_density"),
        "pitch_diversity_mean": (report["metrics"].get("midi") or {}).get("mean_pitch_diversity"),
        "rhythm_diversity_mean": (report["metrics"].get("midi") or {}).get("mean_rhythm_diversity"),
        "tempo_mean": ((report["metrics"].get("tempo") or {}).get("generated") or {}).get("mean"),
        "generated_audio": len(generated_audio),
        "generated_midis": len(generated_midis),
        "genre_summary": (report["metrics"].get("genre_summary") or {}).get("genres", {}),
        "genre_comparison": (report["metrics"].get("genre_summary") or {}).get("comparison", {}),
        "errors": report["errors"],
    }
    _json_dump(metrics_dir / "summary.json", summary)
    report["summary"] = summary

    # Entregables tabulares de la Fase 4 (results/metrics_all.csv + set_level).
    try:
        deliverables = _write_phase4_deliverables(
            evaluation_dir=evaluation_dir,
            report=report,
            summary=summary,
            fad_extractor=fad_extractor,
        )
        report["deliverables"] = deliverables
    except Exception as exc:  # nunca romper el reporte por los CSV
        report["errors"].append({"metric": "deliverables", "error": str(exc)})

    report_path = evaluation_dir / "report.json"
    _json_dump(report_path, report)
    return report


def _write_phase4_deliverables(
    *,
    evaluation_dir: Path,
    report: dict[str, Any],
    summary: dict[str, Any],
    fad_extractor: str,
) -> dict[str, str]:
    """
    Escribe results/metrics_all.csv (por clip) y results/set_level_metrics.csv
    con las columnas del protocolo, a partir de lo ya calculado en el reporte.
    KLD es distribucional (a nivel de conjunto): se repite por fila del modelo.
    """
    from hybrid_music_engine.new_metrics.deliverables import write_deliverables

    results_dir = evaluation_dir / "results"
    model_name = str(report.get("evaluation_id") or evaluation_dir.name)

    # CLAP por clip: mapa audio_path -> similitud.
    clap_by_audio: dict[str, float] = {}
    for entry in (report["metrics"].get("clap") or {}).get("per_pair", []) or []:
        if entry.get("audio") is not None and entry.get("similarity") is not None:
            clap_by_audio[str(entry["audio"])] = float(entry["similarity"])

    set_kld = summary.get("kld")
    clip_rows: list[dict[str, Any]] = []
    for row in (report["metrics"].get("pairs") or {}).get("rows", []) or []:
        row_metrics = row.get("metrics") or {}
        gen_audio = (row.get("generated") or {}).get("audio_path")
        clip_rows.append({
            "model": model_name,
            "clip_id": row.get("track_id") or row.get("pair_id"),
            "genre": row.get("genre"),
            "tempo_bpm": row_metrics.get("generated_tempo") or row_metrics.get("tempo"),
            "clap_score": clap_by_audio.get(str(gen_audio)) if gen_audio else None,
            "passt_kld": set_kld,
        })

    set_row = {
        "model": model_name,
        "fad_vggish": summary.get("fad") if fad_extractor == "vggish" else None,
        "fad_pann": summary.get("fad") if fad_extractor == "pann" else None,
        "mean_clap": summary.get("clap"),
        "std_clap": summary.get("clap_std"),
        "pct_clap_above_025": summary.get("pct_clap_above_025"),
        "kld": summary.get("kld"),
        "kad": summary.get("kad"),
    }
    # El extractor mel (por defecto en la ruta integrada) no es vggish/pann: se
    # deja constancia del valor FAD bajo una clave auxiliar para no perderlo.
    if fad_extractor not in {"vggish", "pann"}:
        set_row["fad_mel"] = summary.get("fad")

    return write_deliverables(clip_rows, [set_row], results_dir)


def list_evaluations(config: EngineConfig) -> list[dict[str, Any]]:
    root = config.data_dir / "evaluations"
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/manifest.json"), reverse=True):
        try:
            manifest = _load_json(path)
        except Exception:
            continue
        report_path = path.parent / "report.json"
        summary_path = path.parent / "metrics" / "summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        rows.append(
            {
                "evaluation_id": manifest.get("evaluation_id", path.parent.name),
                "created_at": manifest.get("created_at"),
                "completed_at": manifest.get("completed_at"),
                "status": manifest.get("status"),
                "source": manifest.get("source"),
                "distribution": manifest.get("distribution", {}),
                "valid_tracks": manifest.get("valid_tracks"),
                "path": str(path),
                "report_path": str(report_path) if report_path.exists() else None,
                "summary": summary,
                "report_download_url": data_file_url(config, report_path) if report_path.exists() else None,
                "manifest_download_url": data_file_url(config, path),
            }
        )
    return rows


def load_evaluation(config: EngineConfig, evaluation_id: str) -> dict[str, Any]:
    path = config.data_dir / "evaluations" / evaluation_id / "manifest.json"
    if not path.exists():
        raise RuntimeError("Evaluación no encontrada.")
    payload = _load_json(path)
    report_path = path.parent / "report.json"
    if report_path.exists():
        payload["report"] = _load_json(report_path)
    return payload


def load_evaluation_report(config: EngineConfig, evaluation_id: str) -> dict[str, Any]:
    path = config.data_dir / "evaluations" / evaluation_id / "report.json"
    if not path.exists():
        raise RuntimeError("Reporte de evaluación no encontrado.")
    return _load_json(path)


def evaluation_files(config: EngineConfig, evaluation_id: str) -> dict[str, Any]:
    root = config.data_dir / "evaluations" / evaluation_id
    if not root.exists():
        raise RuntimeError("Evaluación no encontrada.")
    allowed = {".json", ".csv", ".png", ".jpg", ".jpeg", ".npy", ".wav", ".mp3", ".mid", ".midi"}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            files.append(
                {
                    "label": str(path.relative_to(root)),
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "download_url": data_file_url(config, path),
                }
            )
    return {"evaluation_id": evaluation_id, "files": files}
