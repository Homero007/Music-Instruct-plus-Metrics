#!/usr/bin/env python3
"""
passt_classifier.py — Clasificador PaSST (527 clases AudioSet) para KLD.

PaSST (Patchout faSt Spectrogram Transformer) preentrenado en AudioSet produce,
por clip, un vector de probabilidades p ∈ R^527 sobre clases acústicas. Esas
distribuciones alimentan la métrica KLD del protocolo (kld_metric.py):

  p_real = mean([passt(x) for x en testset])
  p_gen  = mean([passt(x) for x en generado])
  KLD    = Σ p_real · log(p_real / p_gen + ε)

Es un BACKEND OPCIONAL: si `hear21passt` no está instalado, lanza un error claro
con la instrucción de instalación, igual que el resto de extractores del paquete.
El clasificador de centroides propio (audio_classifier/) sigue siendo el camino
por defecto del pipeline integrado; PaSST es la opción "de referencia" del
protocolo cuando se quieren las 527 clases de AudioSet.

Uso como módulo:
    from passt_classifier import PaSSTClassifier
    clf = PaSSTClassifier(device="cpu")
    probs = clf.predict_proba("clip.wav")          # (527,)
    matrix = clf.predict_batch(["a.wav", "b.wav"])  # (N, 527)

Combinado con KLD:
    from kld_metric import promediar_predicciones, calcular_kld_desde_distribuciones
    p = promediar_predicciones(clf.predict_batch(real_files))
    q = promediar_predicciones(clf.predict_batch(gen_files))
    kld = calcular_kld_desde_distribuciones(p, q)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

PASST_SR = 32000           # PaSST espera audio a 32 kHz mono
PASST_N_CLASSES = 527      # taxonomía AudioSet


def _dep_error(package: str, install_cmd: str) -> None:
    print(f"\n[ERROR] Paquete requerido no encontrado: {package}")
    print(f"        Instalar con: {install_cmd}\n")
    sys.exit(1)


class PaSSTClassifier:
    """
    Wrapper sobre hear21passt que expone predict_proba / predict_batch con
    probabilidades de 527 clases AudioSet (softmax sobre los logits del modelo).
    """

    n_classes = PASST_N_CLASSES

    def __init__(self, device: str = "cpu", sr: int = PASST_SR):
        try:
            import torch  # noqa: F401
            import librosa  # noqa: F401
            from hear21passt.base import get_basic_model  # noqa: F401
        except ImportError:
            _dep_error("hear21passt", "pip install hear21passt librosa torch")

        import torch
        import librosa
        from hear21passt.base import get_basic_model

        self.torch = torch
        self.librosa = librosa
        self.sr = sr
        self.device = device
        log.info("Cargando PaSST (AudioSet, 527 clases) en device='%s'…", device)
        self.model = get_basic_model(mode="logits")
        self.model.eval()
        self.model.to(device)

    def _logits(self, audio_path: Path) -> np.ndarray:
        y, _ = self.librosa.load(str(audio_path), sr=self.sr, mono=True)
        tensor = self.torch.tensor(y, dtype=self.torch.float32, device=self.device)[None, :]
        with self.torch.no_grad():
            logits = self.model(tensor)
        return logits.squeeze(0).detach().cpu().numpy().astype(np.float64)

    def predict_proba(self, audio_path: Path) -> np.ndarray:
        """Vector de probabilidades (527,) vía softmax sobre los logits."""
        logits = self._logits(Path(audio_path))
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        total = exp.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(self.n_classes, 1.0 / self.n_classes, dtype=np.float64)
        return exp / total

    def predict_batch(self, audio_paths) -> np.ndarray:
        """Matriz (N, 527) de probabilidades; salta archivos que fallen."""
        rows: list[np.ndarray] = []
        for path in audio_paths:
            try:
                rows.append(self.predict_proba(Path(path)))
            except Exception as exc:  # noqa: BLE001
                log.warning("PaSST: error en %s: %s — omitido", path, exc)
        if not rows:
            return np.empty((0, self.n_classes), dtype=np.float64)
        return np.vstack(rows)


def compute_kld_passt(
    real_files,
    gen_files,
    *,
    device: str = "cpu",
    epsilon: float = 1e-10,
) -> dict:
    """
    Atajo: clasifica con PaSST y calcula KLD(p_real ‖ p_gen) con la matemática
    de kld_metric.py. Devuelve dict con kld y tamaños de muestra.
    """
    try:  # pragma: no cover - depende del contexto de import
        from .kld_metric import calcular_kld_desde_distribuciones, promediar_predicciones
    except ImportError:
        from kld_metric import calcular_kld_desde_distribuciones, promediar_predicciones

    clf = PaSSTClassifier(device=device)
    real_probs = clf.predict_batch(real_files)
    gen_probs = clf.predict_batch(gen_files)
    if real_probs.shape[0] == 0 or gen_probs.shape[0] == 0:
        raise RuntimeError("PaSST no pudo clasificar suficientes clips para KLD.")
    p = promediar_predicciones(real_probs, epsilon=epsilon)
    q = promediar_predicciones(gen_probs, epsilon=epsilon)
    return {
        "kld": calcular_kld_desde_distribuciones(p, q, epsilon=epsilon),
        "backend": "passt",
        "n_classes": PASST_N_CLASSES,
        "real_n": int(real_probs.shape[0]),
        "generated_n": int(gen_probs.shape[0]),
    }
