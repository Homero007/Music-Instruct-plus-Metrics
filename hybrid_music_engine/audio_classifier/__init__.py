"""Lightweight audio classifier used to produce probabilities for KLD."""

from .model import (
    DEFAULT_CLASS_LABELS,
    AudioCentroidClassifier,
    predict_batch,
    predict_proba,
    train,
    train_audio_classifier,
)

class_labels = DEFAULT_CLASS_LABELS

__all__ = [
    "DEFAULT_CLASS_LABELS",
    "AudioCentroidClassifier",
    "class_labels",
    "predict_batch",
    "predict_proba",
    "train",
    "train_audio_classifier",
]
