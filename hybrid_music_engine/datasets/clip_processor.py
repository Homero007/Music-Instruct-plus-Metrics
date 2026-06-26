from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.audio.loader import import_and_normalize_audio
from hybrid_music_engine.audio.stems import separate_stems
from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id
from hybrid_music_engine.features.global_features import extract_midi_features
from hybrid_music_engine.tokens.midi_tokenizer import tokenize_midi_file
from hybrid_music_engine.transcription.drums_onsets import transcribe_drums_onsets
from hybrid_music_engine.transcription.midi_cleanup import cleanup_midi_layer
from hybrid_music_engine.transcription.melodic_basic_pitch import transcribe_melodic_basic_pitch


def process_jamendo_clips(
    config: EngineConfig,
    *,
    clips_catalog_path: Path,
    max_clips: int | None = None,
    run_stems: bool = False,
    run_melodic: bool = True,
    run_drums: bool = True,
    run_features: bool = True,
    run_tokens: bool = True,
    continue_on_error: bool = True,
    processing_mode: str = "quick",
    midi_cleanup: bool = False,
    quantize_grid: str = "1/16",
    strict_demucs: bool = False,
) -> dict[str, Any]:
    if processing_mode not in {"quick", "token_vae_demucs"}:
        raise RuntimeError("processing_mode debe ser 'quick' o 'token_vae_demucs'.")
    serious_mode = processing_mode == "token_vae_demucs"
    if serious_mode:
        run_stems = True
        run_melodic = True
        run_drums = True
        run_features = True
        run_tokens = True
        midi_cleanup = True
        strict_demucs = True

    catalog_file = Path(clips_catalog_path).expanduser().resolve()
    if not catalog_file.exists():
        raise RuntimeError(f"Catálogo de clips no encontrado: {catalog_file}")
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    entries = list(catalog.get("entries", []))
    if max_clips is not None:
        entries = entries[:max_clips]
    if not entries:
        raise RuntimeError("El catálogo de clips no contiene entradas.")

    batch_id = create_id("jamendo_batch", prefix="batch")
    output_root = config.datasets_dir / "jamendo" / str(catalog.get("source_catalog_id", "unknown")) / "processed" / batch_id
    output_root.mkdir(parents=True, exist_ok=True)

    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    token_entries: list[dict[str, Any]] = []

    for entry in entries:
        clip_id = str(entry.get("clip_id") or create_id("clip", prefix="clip"))
        genre = str(entry.get("genre") or "unknown")
        clip_path = Path(str(entry.get("clip_path") or "")).expanduser()
        clip_dir = output_root / genre / clip_id
        source_dir = clip_dir / "source"
        stems_dir = clip_dir / "stems"
        midis_dir = clip_dir / "midis"
        features_dir = clip_dir / "features"
        tokens_dir = clip_dir / "tokens"
        for directory in [source_dir, stems_dir, midis_dir, features_dir, tokens_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        record: dict[str, Any] = {
            "clip_id": clip_id,
            "genre": genre,
            "source_clip": str(clip_path),
            "directory": str(clip_dir),
            "status": "started",
            "errors": [],
            "warnings": [],
        }
        try:
            if not clip_path.exists():
                raise RuntimeError(f"Clip no encontrado: {clip_path}")
            imported = import_and_normalize_audio(clip_path, source_dir, config)
            record["source"] = imported
            normalized = Path(str(imported["normalized"]))

            stems_payload = None
            if run_stems:
                try:
                    stems_payload = separate_stems(normalized, stems_dir)
                    record["stems"] = stems_payload
                except RuntimeError as exc:
                    if strict_demucs or not (run_melodic or run_drums):
                        raise
                    record["warnings"].append(
                        "No se pudieron separar stems; se procesó el audio normalizado directo. "
                        f"Detalle: {exc}"
                    )
                    stems_payload = None

            midi_payloads: dict[str, Any] = {}
            cleanup_payloads: dict[str, Any] = {}
            if run_drums:
                drums_source = normalized
                if stems_payload and stems_payload.get("files", {}).get("drums"):
                    drums_source = Path(str(stems_payload["files"]["drums"]))
                midi_payloads["drums"] = transcribe_drums_onsets(
                    drums_source,
                    midis_dir / "drums.mid",
                )
                if midi_cleanup:
                    cleanup_payloads["drums"] = cleanup_midi_layer(
                        midis_dir / "drums.mid",
                        layer="drums",
                        quantize_grid=quantize_grid,
                        min_note_ticks=24,
                        velocity_floor=20,
                    )
            if run_melodic:
                melodic_sources: dict[str, Path] = {}
                if serious_mode and stems_payload:
                    files = stems_payload.get("files", {})
                    if files.get("bass"):
                        melodic_sources["bass"] = Path(str(files["bass"]))
                    melody_source = _first_useful_stem(files, ["vocals", "other"])
                    if melody_source:
                        melodic_sources["melody"] = melody_source
                    if files.get("other"):
                        melodic_sources["harmony"] = Path(str(files["other"]))
                elif stems_payload:
                    for name in ["bass", "vocals", "other"]:
                        path = stems_payload.get("files", {}).get(name)
                        if path:
                            melodic_sources[name] = Path(str(path))
                else:
                    melodic_sources["melody"] = normalized
                melodic_payloads = {}
                for name, path in melodic_sources.items():
                    melodic_payloads[name] = transcribe_melodic_basic_pitch(
                        path,
                        midis_dir / f"{name}.mid",
                    )
                    if midi_cleanup:
                        cleanup_payloads[name] = cleanup_midi_layer(
                            midis_dir / f"{name}.mid",
                            layer=name if name in {"bass", "melody", "harmony"} else "melody",
                            quantize_grid=quantize_grid,
                            min_note_ticks=96 if name == "harmony" else 48,
                            velocity_floor=14,
                        )
                midi_payloads["melodic"] = melodic_payloads
            record["midis"] = midi_payloads
            if cleanup_payloads:
                record["midi_cleanup"] = cleanup_payloads
                for layer, cleanup in cleanup_payloads.items():
                    if not cleanup.get("valid"):
                        record["warnings"].append(f"La capa {layer} quedó sin notas útiles tras limpieza.")

            feature_payloads = {}
            if run_features:
                for midi_path in sorted(midis_dir.glob("*.mid")):
                    feature_payloads[midi_path.stem] = extract_midi_features(midi_path)
                features_path = features_dir / "features.json"
                features_path.write_text(json.dumps(feature_payloads, indent=2), encoding="utf-8")
                record["features"] = {"path": str(features_path), "items": feature_payloads}

            if run_tokens:
                for midi_path in sorted(midis_dir.glob("*.mid")):
                    token_payload = tokenize_midi_file(midi_path, genre=genre, clip_id=clip_id)
                    token_path = tokens_dir / f"{midi_path.stem}.tokens.json"
                    token_payload["path"] = str(token_path)
                    token_payload["processing_mode"] = processing_mode
                    token_payload["layer"] = midi_path.stem
                    token_path.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
                    token_entries.append(
                        {
                            "clip_id": clip_id,
                            "genre": genre,
                            "path": str(token_path),
                            "source_midi": str(midi_path),
                            "layer": midi_path.stem,
                            "token_count": token_payload["token_count"],
                            "duration_seconds": token_payload["duration_seconds"],
                        }
                    )
                record["tokens_dir"] = str(tokens_dir)

            record["status"] = "completed"
        except Exception as exc:
            record["status"] = "failed"
            record["errors"].append(str(exc))
            failed.append(record)
            if not continue_on_error:
                raise

        manifest_path = clip_dir / "manifest.json"
        record["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        processed.append(record)

    token_manifest = {
        "schema_version": "token-set-manifest-v1",
        "kind": "processed-jamendo-clips",
        "token_set_id": create_id(batch_id, prefix="tokens"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "clips_catalog_path": str(catalog_file),
        "processing_mode": processing_mode,
        "intended_model": "token_vae" if serious_mode else "transformer",
        "total_files": len(token_entries),
        "entries": token_entries,
    }
    token_manifest_path = output_root / "tokens_manifest.json"
    token_manifest["path"] = str(token_manifest_path)
    token_manifest_path.write_text(json.dumps(token_manifest, indent=2), encoding="utf-8")

    batch_manifest_path = output_root / "batch_manifest.json"
    batch_manifest = {
        "schema_version": "jamendo-processed-batch-v1",
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "clips_catalog_path": str(catalog_file),
        "root": str(output_root),
        "total_requested": len(entries),
        "total_completed": sum(1 for item in processed if item["status"] == "completed"),
        "total_failed": len(failed),
        "options": {
            "max_clips": max_clips,
            "run_stems": run_stems,
            "run_melodic": run_melodic,
            "run_drums": run_drums,
            "run_features": run_features,
            "run_tokens": run_tokens,
            "continue_on_error": continue_on_error,
            "processing_mode": processing_mode,
            "midi_cleanup": midi_cleanup,
            "quantize_grid": quantize_grid,
            "strict_demucs": strict_demucs,
        },
        "token_manifest_path": str(token_manifest_path),
        "entries": processed,
        "failed": failed,
        "path": str(batch_manifest_path),
    }
    batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    if run_tokens and not token_entries:
        if serious_mode and failed:
            raise RuntimeError(
                "Token-VAE Demucs no generó tokens. Revisa Demucs/stems/transcripción. "
                f"Primer error: {failed[0]['errors'][0]}"
            )
        raise RuntimeError(
            "No se generaron secuencias de tokens válidas. Activa MIDI melódico o MIDI batería, "
            "procesa más clips, o revisa que los clips seleccionados tengan señal musical suficiente."
        )
    return batch_manifest


def _first_useful_stem(files: dict[str, Any], names: list[str]) -> Path | None:
    for name in names:
        path_value = files.get(name)
        if not path_value:
            continue
        path = Path(str(path_value))
        if _has_signal(path):
            return path
    return None


def _has_signal(path: Path, threshold: float = 0.003) -> bool:
    try:
        import numpy as np
        import soundfile as sf

        data, _sample_rate = sf.read(path, always_2d=False)
        if data.size == 0:
            return False
        return float(np.sqrt(np.mean(np.square(data)))) >= threshold
    except (OSError, RuntimeError, ValueError):
        return path.exists()
