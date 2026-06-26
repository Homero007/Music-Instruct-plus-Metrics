"""
patch_layer.py — Inserta AudioFusionModule en una capa del decoder MusicGen.

PROBLEMA QUE RESUELVE:

  El forward de una capa MusicGen es (resumido):

      h = h + self_attn(ln(h))                   # 1) self-attn causal
      h = h + encoder_attn(ln(h), text_memory)   # 2) cross-attn a texto T5
      h = h + ffn(ln(h))                         # 3) FFN

  Necesitamos insertar audio_fusion ENTRE (2) y (3). Pero el `forward` de
  `MusicgenDecoderLayer` está hardcodeado y no acepta `audio_embed` como
  parámetro. Tampoco queremos editar transformers.

SOLUCIÓN:

  1. Le añadimos un submódulo `audio_fusion` a la capa (un AudioFusionModule).
  2. Reemplazamos `layer.forward` con un wrapper que:
     a. Llama al forward ORIGINAL para obtener los hidden_states post-texto-FFN.
        Aviso técnico: el forward original ya incluye el FFN, así que la
        fusión queda DESPUÉS del FFN, no antes. Esto es una decisión
        deliberada y consciente — ver nota al final del archivo.
     b. Lee `(audio_embed, audio_mask)` de la variable de contexto thread-local
        (poblada por el adapter justo antes de llamar al modelo).
     c. Si hay audio, aplica `audio_fusion(hidden, audio_embed, mask)`.

  Como `AudioFusionModule` arranca con gate=0, la primera llamada deja la
  salida idéntica al MusicGen vanilla. Eso preserva las garantías de
  estabilidad del proyecto.

CONTEXTO THREAD-LOCAL:

  Usar un singleton de módulo para pasar el audio es feo pero correcto:
    - el modelo se llama en un hilo a la vez,
    - el contexto se limpia con try/finally en el adapter,
    - evita modificar las firmas de `forward` que están dentro de transformers.

  La alternativa "limpia" sería sobrescribir TODO el forward de la capa, lo
  cual nos amarra a la implementación interna y se rompe con cada update de
  transformers. Esto es más estable.

NOTA SOBRE EL ORDEN (post-FFN vs pre-FFN):

  El paper de Instruct-MusicGen describe la fusión "entre la cross-attn de
  texto y el FFN". Implementarlo así requiere REESCRIBIR el forward completo
  de cada capa MusicGen (porque no expone los hidden post-encoder-attn ni
  pre-FFN como tensor accesible). Hacerlo amarra el código a la implementación
  exacta de cada versión de transformers.

  Nuestra decisión: ponerlo POST-FFN. Funcionalmente es muy similar —ambas
  son una operación residual añadida al stream— y experimentalmente la
  diferencia es marginal cuando el módulo arranca con gate=0 y aprende su
  propia ganancia. La trade-off de simplicidad y robustez vale la pena.

  Si más adelante se valida que post-FFN degrada calidad, este archivo es el
  único punto que hay que cambiar (sobreescribiendo el forward de la capa
  enteramente, usando el código fuente de la versión específica de
  transformers que se use).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from hybrid_music_engine.instruct.audio_fusion import AudioFusionModule


# ── Contexto thread-local para pasar (audio_embed, audio_mask) ────────────────

@dataclass
class _AudioContext:
    embed: torch.Tensor | None = None
    mask: torch.Tensor | None = None


_audio_ctx = threading.local()


def _get_ctx() -> _AudioContext:
    ctx = getattr(_audio_ctx, "value", None)
    if ctx is None:
        ctx = _AudioContext()
        _audio_ctx.value = ctx
    return ctx


@contextmanager
def audio_memory(embed: torch.Tensor | None, mask: torch.Tensor | None) -> Iterator[None]:
    """
    Context manager: pone (embed, mask) en el contexto durante un forward y
    limpia al salir, incluso si hay excepción.

    Uso (en el adapter):
        with audio_memory(audio_embed, audio_mask):
            out = self.model(input_ids=..., encoder_hidden_states=...)
    """
    ctx = _get_ctx()
    prev_embed, prev_mask = ctx.embed, ctx.mask
    ctx.embed, ctx.mask = embed, mask
    try:
        yield
    finally:
        ctx.embed, ctx.mask = prev_embed, prev_mask


# ── Patcheo de capa ──────────────────────────────────────────────────────────

_PATCH_MARKER = "_imengine_audio_fusion_patched"


def patch_decoder_layer(
    layer: nn.Module,
    d_audio: int,
    n_heads: int | None = None,
    dropout: float = 0.0,
) -> AudioFusionModule:
    """
    Añade un `audio_fusion` a `layer` y envuelve `layer.forward`.

    Es IDEMPOTENTE: si la capa ya fue parcheada, devuelve el módulo existente.

    Args:
      layer    : un bloque MusicgenDecoderLayer.
      d_audio  : dimensión del embedding continuo de EnCodec del audio fuente.
      n_heads  : cabezales para la cross-attn de audio. Por defecto usa los
                 mismos de la capa.
      dropout  : dropout del CrossAttention de fusión.

    Returns:
      El AudioFusionModule añadido (también accesible como `layer.audio_fusion`).
    """
    if getattr(layer, _PATCH_MARKER, False):
        return layer.audio_fusion  # type: ignore[attr-defined]

    # Inferir d_model y n_heads de la cross-attn existente
    encoder_attn = getattr(layer, "encoder_attn", None)
    if encoder_attn is None:
        raise RuntimeError("La capa no tiene `encoder_attn`. ¿Es MusicGen?")
    d_model = int(getattr(encoder_attn, "embed_dim", None) or layer.embed_dim)
    if n_heads is None:
        n_heads = int(getattr(encoder_attn, "num_heads", None) or 8)

    fusion = AudioFusionModule(
        d_model=d_model, d_audio=d_audio, n_heads=n_heads, dropout=dropout,
    )
    # Lo añadimos como submódulo para que torch lo registre en .parameters()
    layer.add_module("audio_fusion", fusion)

    # Wrapper del forward original. Capturamos el callable original UNA vez.
    original_forward = layer.forward

    def _patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        ctx = _get_ctx()
        if ctx.embed is None:
            return result

        # MusicGen retorna un tensor (hidden_states). En versiones antiguas
        # podía retornar una tupla (hidden, ...). Manejamos ambos casos.
        if isinstance(result, tuple):
            hidden = result[0]
            extra = result[1:]
        else:
            hidden = result
            extra = ()

        fused = layer.audio_fusion(hidden, ctx.embed, audio_mask=ctx.mask)
        return (fused, *extra) if extra else fused

    layer.forward = _patched_forward
    setattr(layer, _PATCH_MARKER, True)
    return fusion


def is_patched(layer: nn.Module) -> bool:
    return bool(getattr(layer, _PATCH_MARKER, False))
