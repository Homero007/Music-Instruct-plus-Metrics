"""
audio_encoder.py — EnCodec: tokens RVQ discretos + embedding continuo.

El audio original a editar se codifica con EnCodec. Produce dos cosas:

  codes      : (n_codebooks, n_frames) int  — tokens discretos que el decoder
                                              autorregresivo predice.
  embeddings : (n_frames, dim)        float — latente continuo que alimenta el
                                              AudioFusionModule como "memoria".

Corrige un detalle del script original: en transformers, `model.encoder(x)`
devuelve un TENSOR (B, dim, frames), no un objeto con `.last_hidden_state`.
Aquí lo manejamos de forma robusta y lo transponemos a (frames, dim).

NOTA DE COMPATIBILIDAD con MusicGen:
  MusicGen consume tokens de `facebook/encodec_32khz` (4 codebooks). El script
  original usa `facebook/encodec_24khz` (8 codebooks @ 3 kbps). Para conectar
  con un checkpoint MusicGen real, usa MUSICGEN_ENCODEC abajo. Para experimentos
  propios independientes, 24 kHz funciona igual.

Requiere: transformers, torch, librosa.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MUSICGEN_ENCODEC = "facebook/encodec_32khz"   # 4 codebooks — compatible con MusicGen
DEFAULT_ENCODEC = "facebook/encodec_32khz"


class EnCodecAudioEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_ENCODEC,
        bandwidth: float | None = None,
        device: str = "cpu",
    ):
        import torch
        from transformers import AutoProcessor, EncodecModel

        self.device = torch.device(device)
        self.bandwidth = bandwidth
        log.info("Cargando EnCodec: %s …", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = EncodecModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.target_sr = int(self.processor.sampling_rate)
        for param in self.model.parameters():
            param.requires_grad = False
        log.info("EnCodec listo (congelado). SR=%d Hz", self.target_sr)

    def _continuous_embedding(self, input_values):
        """Devuelve el latente continuo (B, dim, frames) de forma robusta."""
        enc = self.model.encoder(input_values)
        # En transformers el encoder retorna un Tensor; algunas versiones podrían
        # envolverlo. Cubrimos ambos casos.
        return getattr(enc, "last_hidden_state", enc)

    def encode_file(self, audio_path: Path) -> dict:
        import librosa
        import torch

        y, _ = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        duration = len(y) / self.target_sr

        inputs = self.processor(
            raw_audio=y,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            kwargs = {}
            if self.bandwidth is not None:
                kwargs["bandwidth"] = self.bandwidth
            encoded = self.model.encode(
                inputs["input_values"], inputs.get("padding_mask"), **kwargs
            )
            codes = encoded.audio_codes[0].cpu().numpy()            # (n_codebooks, n_frames)

            latent = self._continuous_embedding(inputs["input_values"])
            embeddings = latent[0].transpose(0, 1).cpu().numpy()    # (n_frames, dim)

        return {"codes": codes, "embeddings": embeddings, "duration": duration}


# ── Helpers de carga (coherentes con encode_audio_text.py original) ───────────

def load_encodec_codes(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Códigos EnCodec no encontrados: {path}")
    return np.load(path)


def load_encodec_embedding(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embedding EnCodec no encontrado: {path}")
    return np.load(path)
