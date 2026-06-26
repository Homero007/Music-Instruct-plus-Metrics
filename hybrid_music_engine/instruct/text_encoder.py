"""
text_encoder.py — Codificador T5 que preserva la secuencia COMPLETA.

Punto 2 de la estrategia: el script original colapsaba la salida de T5 a un único
vector (768,) mediante mean-pooling. Eso es un cuello de botella: pierde qué token
corresponde a "Trompeta", cuál a "120 BPM" y cuál al verbo de acción "Remove".

Aquí extraemos `last_hidden_state` completo con forma (L, 768) y su máscara de
atención. Esta es exactamente la representación que MusicGen / Instruct-MusicGen
consumen por cross-attention: NUNCA hacen pooling del texto.

Se conserva `encode_pooled()` por compatibilidad con código antiguo que esperaba
un vector por género, pero el camino principal es `encode_sequence()`.

Requiere: transformers, torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

T5_MODEL = "google-t5/t5-base"   # 768 dim, el mismo que usa MusicGen
T5_DIM = 768


@dataclass
class EncodedText:
    """
    Resultado de codificar uno o varios textos.

    hidden : (B, L, D) float32 — estados ocultos de la última capa del encoder T5
    mask   : (B, L)    int     — 1 = token real, 0 = padding
    texts  : lista de los textos originales (para trazabilidad)

    Para un solo texto, usa `.single()` para obtener (L, D) y (L,) sin padding.
    """

    hidden: np.ndarray
    mask: np.ndarray
    texts: list[str]

    def single(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (L_real, D) y (L_real,) del primer (y único) elemento, sin padding."""
        valid = self.mask[0].astype(bool)
        return self.hidden[0][valid], self.mask[0][valid]

    def unpadded(self) -> list[np.ndarray]:
        """Lista de tensores (L_i, D) por elemento, recortando el padding de cada uno."""
        out = []
        for h, m in zip(self.hidden, self.mask):
            out.append(h[m.astype(bool)])
        return out


class T5SequenceEncoder:
    """
    Codifica texto con el encoder de T5 y devuelve la secuencia completa.

    A diferencia de T5TextEncoder original, NO promedia: retorna (L, D).
    """

    def __init__(self, model_name: str = T5_MODEL, device: str = "cpu", max_length: int = 128):
        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = torch.device(device)
        self.max_length = max_length
        log.info("Cargando T5 encoder (secuencia completa): %s …", model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        # Congelamos T5: en Instruct-MusicGen el codificador de texto NO se entrena.
        for param in self.model.parameters():
            param.requires_grad = False
        log.info("T5 listo y congelado. Dimensión: %d", self.model.config.d_model)

    @property
    def dim(self) -> int:
        return int(self.model.config.d_model)

    def encode_sequence(self, text: str | list[str]) -> EncodedText:
        """
        Codifica texto preservando la secuencia.

        Retorna EncodedText con hidden (B, L, D) y mask (B, L).
        """
        import torch

        texts = [text] if isinstance(text, str) else list(text)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state  # (B, L, D)

        return EncodedText(
            hidden=hidden.cpu().float().numpy(),
            mask=inputs["attention_mask"].cpu().numpy(),
            texts=texts,
        )

    def encode_pooled(self, text: str | list[str]) -> np.ndarray:
        """
        COMPATIBILIDAD: mean-pooling como el script original.

        Útil solo si algún consumidor antiguo necesita un vector (D,) por texto.
        El camino recomendado es encode_sequence().
        """
        enc = self.encode_sequence(text)
        mask = enc.mask[..., None].astype(np.float32)        # (B, L, 1)
        pooled = (enc.hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        return pooled[0] if isinstance(text, str) else pooled


# ── Persistencia de secuencias (caché por pista) ──────────────────────────────
#
# Como L varía por texto, guardamos cada secuencia en su propio .npz (sin padding).
# Almacenamos en float16 para ahorrar disco; T5 condiciona bien en esa precisión.

def save_sequence(path: Path, hidden: np.ndarray, text: str) -> None:
    """Guarda una secuencia (L, D) + el texto fuente en un .npz comprimido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hidden=hidden.astype(np.float16),
        text=np.array(text),
        shape=np.array(hidden.shape),
    )


def load_sequence(path: Path) -> tuple[np.ndarray, str]:
    """Carga una secuencia (L, D) float32 + su texto."""
    if not path.exists():
        raise FileNotFoundError(f"Secuencia T5 no encontrada: {path}")
    data = np.load(path, allow_pickle=False)
    return data["hidden"].astype(np.float32), str(data["text"])


def encode_and_cache(
    encoder: T5SequenceEncoder,
    items: list[tuple[str, str]],
    out_dir: Path,
) -> dict[str, Path]:
    """
    Codifica una lista de (clave, texto) y guarda cada secuencia en out_dir/<clave>.npz.

    `clave` se usa como nombre de archivo (sanitízala antes si viene de rutas).
    Retorna {clave: ruta_guardada}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for key, text in items:
        enc = encoder.encode_sequence(text)
        hidden, _ = enc.single()                 # (L, D) sin padding
        path = out_dir / f"{key}.npz"
        save_sequence(path, hidden, text)
        saved[key] = path
        log.info("  T5 seq [%s] → %s  shape=%s", key, path.name, hidden.shape)
    return saved
