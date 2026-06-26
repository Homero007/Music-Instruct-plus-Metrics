"""
musicgen_real — Integración con MusicGen real de HuggingFace.

Reemplaza el `musicgen_adapter.py` (stub) por una integración funcional con
`transformers.MusicgenForCausalLM`. Inyecta `AudioFusionModule` + LoRA en las
cross-attentions del decoder real y entrena con teacher forcing respetando
el delay pattern propio de MusicGen.

Submódulos:
  locate           Localiza las piezas del decoder (robusto a versiones).
  patch_layer      Monkey-patch que añade AudioFusionModule a una capa.
  delay_pattern    Aplica el patrón escalonado de MusicGen a los targets.
  adapter          MusicGenInstructAdapter: orquesta carga → patch → LoRA.
"""

from __future__ import annotations

from .adapter import MusicGenAdapterConfig, MusicGenInstructAdapter
from .delay_pattern import LABEL_IGNORE, prepare_teacher_forcing_inputs
from .locate import DecoderRefs, cross_attn_modules, locate_decoder
from .patch_layer import audio_memory, is_patched, patch_decoder_layer

__all__ = [
    "MusicGenAdapterConfig", "MusicGenInstructAdapter",
    "DecoderRefs", "locate_decoder", "cross_attn_modules",
    "patch_decoder_layer", "audio_memory", "is_patched",
    "prepare_teacher_forcing_inputs", "LABEL_IGNORE",
]
