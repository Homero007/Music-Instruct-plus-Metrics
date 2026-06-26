from __future__ import annotations

from typing import Any


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "label": "Prueba rápida",
        "model_type": "markov",
        "order": 2,
        "epochs": 1,
        "sequence_length": 64,
        "batch_size": 8,
        "embedding_dim": 64,
        "num_layers": 1,
        "num_heads": 2,
    },
    "small": {
        "label": "Transformer pequeño",
        "model_type": "transformer",
        "epochs": 8,
        "sequence_length": 128,
        "batch_size": 16,
        "embedding_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
    },
    "medium": {
        "label": "Transformer medio",
        "model_type": "transformer",
        "epochs": 24,
        "sequence_length": 256,
        "batch_size": 16,
        "embedding_dim": 192,
        "num_layers": 4,
        "num_heads": 6,
    },
    "final": {
        "label": "Entrenamiento serio",
        "model_type": "transformer",
        "epochs": 80,
        "sequence_length": 512,
        "batch_size": 12,
        "embedding_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
    },
    "research": {
        "label": "Transformer profundo",
        "model_type": "transformer",
        "epochs": 160,
        "sequence_length": 768,
        "batch_size": 8,
        "embedding_dim": 384,
        "num_layers": 8,
        "num_heads": 8,
    },
    "production": {
        "label": "Producción larga",
        "model_type": "transformer",
        "epochs": 240,
        "sequence_length": 1024,
        "batch_size": 6,
        "embedding_dim": 512,
        "num_layers": 10,
        "num_heads": 8,
    },
}


GENERATION_PRESETS: dict[str, dict[str, Any]] = {
    "draft_15s": {
        "label": "Borrador 15s",
        "duration_seconds": 15,
        "max_tokens": 512,
        "export_layers": True,
        "feature_tokens": ["density:medium", "duration:short"],
    },
    "full_30s": {
        "label": "Pista 30s",
        "duration_seconds": 30,
        "max_tokens": 1024,
        "temperature": 0.82,
        "top_k": 48,
        "top_p": 0.92,
        "export_layers": True,
        "feature_tokens": ["density:medium", "duration:medium"],
    },
    "dense_30s": {
        "label": "Energía alta 30s",
        "duration_seconds": 30,
        "max_tokens": 1400,
        "temperature": 0.9,
        "top_k": 64,
        "top_p": 0.94,
        "export_layers": True,
        "feature_tokens": ["density:high", "duration:medium"],
    },
    "ambient_60s": {
        "label": "Ambient 60s",
        "duration_seconds": 60,
        "max_tokens": 1600,
        "temperature": 0.72,
        "top_k": 40,
        "top_p": 0.9,
        "export_layers": True,
        "feature_tokens": ["density:low", "duration:long"],
    },
    "fusion_compare_30s": {
        "label": "Comparar fusiones 30s",
        "duration_seconds": 30,
        "max_tokens": 1200,
        "temperature": 0.84,
        "top_k": 56,
        "top_p": 0.92,
        "export_layers": True,
        "feature_tokens": ["density:medium", "duration:medium", "fusion:balanced"],
    },
}


def presets_payload() -> dict[str, Any]:
    return {
        "training": TRAINING_PRESETS,
        "generation": GENERATION_PRESETS,
    }
