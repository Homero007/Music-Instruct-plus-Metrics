"""
reward_model — Modelo de recompensa aprendido sobre features musicales.

Reemplaza heurísticas de ranking por una red entrenada con pérdida pairwise
(Bradley-Terry). Diseñado para integrarse sin invasividad:

  • Lee `data/ranked/<id>/ranking.json` que ya escribe tu pipeline.
  • Escribe `ranking_reranked.json` al lado, conservando el original.
  • Puede arrancarse SIN labels humanos via bootstrap real-vs-generado.

Uso típico (CLI):

  # 1. Bootstrap inicial (sin labels humanos)
  python -m instruct_music_engine.reward_model.cli bootstrap \\
      --real-dir      data/datasets/jamendo/CATALOG/processed/BATCH/midis \\
      --generated-dir data/ranked \\
      --output        data/reward/bootstrap.jsonl

  # 2. Entrenar
  python -m instruct_music_engine.reward_model.cli train \\
      --pairs      data/reward/bootstrap.jsonl \\
      --output-dir data/models/reward/v1

  # 3. Re-rankear todo lo existente
  python -m instruct_music_engine.reward_model.cli rerank-all \\
      --model  data/models/reward/v1/reward_model.pt \\
      --schema data/models/reward/v1/schema.json
"""

from __future__ import annotations

from .features import FeatureSchema
from .model import RewardMLP, RewardModelConfig, bradley_terry_loss, pairwise_accuracy
from .dataset import PreferencePair, read_preference_manifest, build_bootstrap_pairs, merge_pairs

__all__ = [
    "FeatureSchema",
    "RewardMLP", "RewardModelConfig", "bradley_terry_loss", "pairwise_accuracy",
    "PreferencePair", "read_preference_manifest", "build_bootstrap_pairs", "merge_pairs",
]
