"""
locate.py — Localiza las piezas del decoder de MusicGen en HuggingFace.

La estructura interna de `transformers.MusicgenForCausalLM` no es estable
entre versiones (se añadió `attn_implementation`, `Cache`, etc.). En vez de
acceder por path fijo en todo el código, centralizamos aquí los accesos.

Si tu versión de transformers cambia la estructura, debería bastar con
ajustar este archivo y el resto del adapter no necesita cambios.

Probado contra `transformers==5.9.0` y compatible con MusicGen >= 4.31.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class DecoderRefs:
    """Referencias a piezas internas del modelo MusicGen.

    Atributos:
      model_cls    : la clase del modelo cargado (MusicgenForCausalLM o
                     MusicgenForConditionalGeneration)
      decoder      : objeto MusicgenDecoder (con .layers, .embed_tokens, ...)
      layers       : ModuleList con los bloques transformer
      hidden_size  : dimensión del modelo (necesaria para AudioFusionModule)
      num_heads    : cabezales de atención (debe coincidir en la fusión)
      num_codebooks: cuántos codebooks RVQ usa el modelo (4 en MusicGen)
      lm_heads     : ModuleList de cabezales de salida (uno por codebook)
      pad_token_id : id del token de padding/inicio (usado por delay pattern)
    """

    model: nn.Module
    decoder: nn.Module
    layers: nn.ModuleList
    hidden_size: int
    num_heads: int
    num_codebooks: int
    lm_heads: nn.ModuleList
    pad_token_id: int


def locate_decoder(model: nn.Module) -> DecoderRefs:
    """
    Encuentra las piezas relevantes en un modelo MusicGen ya construido.

    Acepta tanto `MusicgenForCausalLM` (decoder solo, usado en fine-tuning)
    como `MusicgenForConditionalGeneration` (incluye encoder T5 y EnCodec
    end-to-end).
    """
    # Caso 1: MusicgenForCausalLM expone .model.decoder directamente.
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        lm_heads = getattr(model, "lm_heads", None)
    # Caso 2: MusicgenForConditionalGeneration anida un nivel más.
    elif hasattr(model, "decoder") and hasattr(model.decoder, "model"):
        decoder = model.decoder.model.decoder
        lm_heads = getattr(model.decoder, "lm_heads", None)
    else:
        raise RuntimeError(
            "No se reconoce la estructura del modelo. ¿Es un MusicgenForCausalLM "
            "o MusicgenForConditionalGeneration?"
        )

    if not hasattr(decoder, "layers") or not isinstance(decoder.layers, nn.ModuleList):
        raise RuntimeError("El decoder no expone `layers: ModuleList` como se esperaba.")

    cfg = getattr(decoder, "config", None) or getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("No pude leer la config del decoder.")

    # MusicgenDecoderConfig usa hidden_size; algunas variantes usan d_model.
    hidden_size = int(getattr(cfg, "hidden_size", None) or cfg.d_model)
    num_heads = int(getattr(cfg, "num_attention_heads", None) or cfg.encoder_attention_heads)
    num_codebooks = int(getattr(cfg, "num_codebooks", 4))
    pad_token_id = int(getattr(cfg, "pad_token_id", 2048) or 2048)

    if lm_heads is None:
        raise RuntimeError("No encontré los lm_heads. ¿El modelo no es MusicGen para LM?")

    return DecoderRefs(
        model=model,
        decoder=decoder,
        layers=decoder.layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_codebooks=num_codebooks,
        lm_heads=lm_heads,
        pad_token_id=pad_token_id,
    )


def cross_attn_modules(refs: DecoderRefs) -> list[Any]:
    """Lista de `encoder_attn` por capa (la cross-attention al texto)."""
    out: list[Any] = []
    for layer in refs.layers:
        if not hasattr(layer, "encoder_attn"):
            raise RuntimeError(
                f"Capa {type(layer).__name__} no tiene `encoder_attn`. "
                f"¿Es realmente un bloque MusicGen?"
            )
        out.append(layer.encoder_attn)
    return out
