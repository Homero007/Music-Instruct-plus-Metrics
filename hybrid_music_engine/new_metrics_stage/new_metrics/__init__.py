"""
new_metrics — Etapa de MÉTRICAS NUEVAS (post-render).

Esta etapa es INDEPENDIENTE del reward model del backend. El reward model
(features.py / model.py / rerank.py) re-rankea candidatas ANTES del render.
Esta etapa corre DESPUÉS de WAV/MP3 y su objetivo es OBTENER GRÁFICOS de
evaluación a partir de las nuevas métricas:

    ... -> reward model (re-ranking) -> WAV/MP3 -> [new_metrics] -> descarga

Contiene (sin alterar su naturaleza):
  • fad_score.py       — Fréchet Audio Distance
  • clap_score.py      — CLAP Score (alineación texto-audio)
  • kld_metric.py      — Kullback-Leibler Divergence
  • evaluate_models.py — comparación multi-modelo + Friedman + xlsx + plots
  • genre_tsne.py      — primitivas t-SNE (librería)
  • words_tsne.py      — t-SNE SOLO de palabras (tokens T5) por género

Orquestador:
  • pipeline.run_new_metrics(...) — genera el paquete de gráficos + report.json
    en data/new_metrics/<run_id>/.

Salida (subcarpeta propia, distinta del reward model):
  data/new_metrics/<run_id>/
    plots/                 ← todos los gráficos (.png)
    audio/                 ← xlsx + caché de evaluate_models
    report.json            ← resumen numérico + rutas de gráficos
"""

from __future__ import annotations

import os
import sys

# Permite que los scripts hermanos (que usan `from fad_score import ...`,
# `from genre_tsne import ...`) se resuelvan sin modificarlos.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = ["run_new_metrics", "run_after_render"]


def __getattr__(name):
    if name == "run_new_metrics":
        from .pipeline import run_new_metrics
        return run_new_metrics
    if name == "run_after_render":
        from .integration import run_after_render
        return run_after_render
    raise AttributeError(name)
