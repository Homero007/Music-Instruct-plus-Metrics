"""
score.py — Inferencia: dict de features → score escalar.

Wrapper limpio sobre el modelo entrenado. Carga modelo y schema una sola vez
y expone `score(metrics)` y `score_batch(metrics_list)`.

Requiere torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from .features import FeatureSchema
from .model import RewardMLP, load_model


class RewardScorer:
    """
    Mantén una instancia única en memoria por proceso para evitar recargar
    el modelo en cada llamada a la API/CLI.
    """

    def __init__(self, model_path: Path | str, schema_path: Path | str, device: str = "cpu"):
        self.device = device
        self.schema = FeatureSchema.from_json(schema_path)
        self.model: RewardMLP = load_model(str(model_path), device=device)
        if self.model.cfg.in_dim != self.schema.dim:
            raise ValueError(
                f"Incompatibilidad modelo↔schema: in_dim={self.model.cfg.in_dim} vs schema.dim={self.schema.dim}. "
                f"Probablemente cargaste un schema diferente al usado en entrenamiento."
            )

    @torch.inference_mode()
    def score(self, metrics: Mapping) -> float:
        x = self.schema.standardize(self.schema.vectorize(metrics))[None, :]
        t = torch.from_numpy(x).to(self.device)
        return float(self.model(t).item())

    @torch.inference_mode()
    def score_batch(self, batch: Iterable[Mapping]) -> np.ndarray:
        X = self.schema.standardize(self.schema.vectorize_batch(batch))
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        t = torch.from_numpy(X).to(self.device)
        return self.model(t).cpu().numpy().astype(np.float32)
