#!/usr/bin/env python3
"""
integration.py — Disparo AUTOMÁTICO de la etapa de métricas nuevas tras el render.

Se invoca desde el flujo de generación (generation/ranked.py) justo después de
renderizar, igual que el re-ranking automático del reward model. Es no fatal y
configurable por entorno.

Contrato mínimo: solo necesita `config.data_dir` (EngineConfig ya lo tiene).

  renders   = config.data_dir / "renders" / run_id        (entrada: WAV/MP3)
  salida    = config.data_dir / "new_metrics" / run_id     (gráficos + report.json)

Variables de entorno
--------------------
  HYBRID_NEW_METRICS_AUTO   "1" (def) corre automático; "0" lo desactiva.
  HYBRID_NEW_METRICS_REAL   carpeta de audio real para FAD (opcional). Sin esto,
                            FAD se omite y se calcula tempo (reference-free).
  HYBRID_T5_DIR             encodings_v2/t5_seq para el t-SNE de palabras (opcional).
  HYBRID_FAD_EXTRACTOR      "mel" (def, sin checkpoint) | "vggish" | "clap".

KLD no se dispara automáticamente: requiere un clasificador que aún no existe.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("hybrid_music_engine.new_metrics")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


def _has_audio(folder: Path) -> bool:
    return folder.exists() and any(
        p.suffix.lower() in AUDIO_EXTENSIONS for p in folder.rglob("*") if p.is_file()
    )


def run_after_render(
    config,
    run_id: str,
    *,
    condition_genre: str | None = None,
    real_root: str | Path | None = None,
    t5_dir: str | Path | None = None,
) -> dict | None:
    """
    Ejecuta la etapa de métricas nuevas sobre los renders de `run_id`.

    Devuelve el report (dict) o None si no había nada que evaluar o está desactivado.
    Nunca lanza: registra y devuelve None ante cualquier fallo (no rompe el job).
    """
    if os.environ.get("HYBRID_NEW_METRICS_AUTO", "1") != "1":
        return None

    try:
        from .pipeline import run_new_metrics

        renders_dir = Path(config.data_dir) / "renders" / run_id
        if not _has_audio(renders_dir):
            log.info("new_metrics: sin audio en %s, se omite.", renders_dir)
            return None

        real_root = real_root or os.environ.get("HYBRID_NEW_METRICS_REAL")
        t5_dir = t5_dir or os.environ.get("HYBRID_T5_DIR")
        fad_extractor = os.environ.get("HYBRID_FAD_EXTRACTOR", "mel")

        # tempo siempre; FAD solo si hay referencia real. KLD nunca (sin clasificador).
        metrics = ["tempos_std"]
        if real_root and _has_audio(Path(real_root)):
            metrics = ["fad", "tempos_std"]
        elif real_root:
            log.warning("new_metrics: HYBRID_NEW_METRICS_REAL sin audio (%s); FAD omitido.", real_root)

        out_dir = Path(config.data_dir) / "new_metrics" / run_id

        report = run_new_metrics(
            generated_root=renders_dir,
            out_dir=out_dir,
            real_root=real_root if (real_root and _has_audio(Path(real_root))) else None,
            t5_dir=t5_dir,
            metrics=metrics,
            fad_extractor=fad_extractor,
            model_label=condition_genre or "generado",
            run_name=run_id,
        )
        log.info("new_metrics: %d gráficos en %s",
                 len(report.get("plots", [])), out_dir)
        return report

    except BaseException as exc:  # noqa: BLE001
        log.warning("new_metrics automático omitido o fallido: %s", exc)
        return None
