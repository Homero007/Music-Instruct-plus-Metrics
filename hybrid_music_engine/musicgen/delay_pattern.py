"""
delay_pattern.py — Alineación de targets con el delay pattern de MusicGen.

CONTEXTO:

  MusicGen no procesa los 4 codebooks alineados en el mismo paso temporal.
  Cada codebook k está retardado k pasos respecto al codebook 0. Para una
  secuencia de 4 codebooks y longitud 8, el patrón se ve así (P=pad, x=token):

      cb 0:  [P, x, x, x, x, P, P, P]
      cb 1:  [P, P, x, x, x, x, P, P]
      cb 2:  [P, P, P, x, x, x, x, P]
      cb 3:  [P, P, P, P, x, x, x, x]

  Ignorar esto y entrenar como si los codebooks fueran síncronos produce un
  modelo que entrena (la loss baja) pero genera audio degradado, porque la
  estructura temporal aprendida no coincide con la del decoder EnCodec.

  Felizmente, `MusicgenForCausalLM` ya expone `build_delay_pattern_mask` que
  hace exactamente esto. Solo necesitamos usarlo para preparar `input_ids` y
  `labels` correctamente para teacher forcing.

LO QUE ESTE MÓDULO HACE:

  prepare_teacher_forcing_inputs(codes, refs)
    Toma codes (B, K, T) ya alineados (síncronos) y devuelve:
      - delayed_input_ids : (B, K, T+K-1)  los inputs con padding aplicado
      - labels            : (B, K, T+K-1)  los targets, con -100 en posiciones de padding

  El modelo entrenará prediciendo cada token desplazado un paso (teacher
  forcing). La cross-entropy con `ignore_index=-100` automáticamente
  descontará todas las posiciones de delay/pad.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .locate import DecoderRefs
except ImportError:
    from locate import DecoderRefs  # type: ignore


# Convención: -100 = ignorar en F.cross_entropy
LABEL_IGNORE = -100


def prepare_teacher_forcing_inputs(
    codes: torch.Tensor,
    refs: DecoderRefs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepara `input_ids` retardados y `labels` para entrenamiento autoregresivo.

    Args:
      codes: (B, K, T) tokens objetivo en formato SÍNCRONO (los codebooks
             alineados en el mismo paso temporal). Es el formato natural que
             produce EnCodec.
      refs:  resultado de `locate_decoder(model)`.

    Returns:
      input_ids: (B * K, T_d)  tokens retardados, listos para `model.forward`.
                 Nota: MusicGen aplana batch y codebook en la primera dim.
      labels:    (B * K, T_d)  targets con LABEL_IGNORE donde no debe predecir.
                 La pérdida estándar de transformers ya respeta esto.

    Donde T_d = T + K (las posiciones extra son por el delay).
    """
    if codes.dim() != 3:
        raise ValueError(f"codes debe ser (B, K, T), recibí {tuple(codes.shape)}")
    B, K, T = codes.shape
    if K != refs.num_codebooks:
        raise ValueError(
            f"codes tiene {K} codebooks pero el modelo espera {refs.num_codebooks}"
        )
    pad_id = refs.pad_token_id

    # IMPORTANTE: el método `build_delay_pattern_mask` de transformers está
    # pensado para INFERENCIA (espera placeholders -1 y trunca al primer -1).
    # Para entrenamiento construimos el patrón explícitamente.
    #
    # Patrón de salida `(B*K, T+K)` con el delay aplicado:
    #
    #   posición 0..K-1+T-1  →  cada codebook k copia su valor desde codes[:, k, :]
    #                            empezando en columna k (offset = k pasos)
    #   resto                 →  pad
    #
    # Ejemplo K=4, T=6 → ancho T_d = T+K = 10
    #
    #     cb 0: [P, x0, x1, x2, x3, x4, x5,  P,  P,  P]
    #     cb 1: [P,  P, y0, y1, y2, y3, y4, y5,  P,  P]
    #     cb 2: [P,  P,  P, z0, z1, z2, z3, z4, z5,  P]
    #     cb 3: [P,  P,  P,  P, w0, w1, w2, w3, w4, w5]
    #
    # La primera posición de cb 0 también es pad porque MusicGen usa un BOS
    # de padding en t=0 (es lo que se ve en build_delay_pattern_mask de HF).
    T_d = T + K
    delayed = torch.full((B, K, T_d), pad_id, dtype=codes.dtype, device=codes.device)
    for k in range(K):
        delayed[:, k, k : k + T] = codes[:, k]
    delayed_flat = delayed.reshape(B * K, T_d)

    # input_ids: lo mismo que delayed_flat (los padding ya están).
    # labels: igual pero con LABEL_IGNORE en posiciones de pad.
    input_ids = delayed_flat.clone()
    labels = delayed_flat.clone()
    labels[labels == pad_id] = LABEL_IGNORE

    return input_ids, labels


def causal_shift_for_labels(
    input_ids: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hace el shift estándar de teacher forcing: input[..., :-1] predice
    labels[..., 1:]. Útil si vamos a calcular la pérdida nosotros mismos
    en vez de delegar en el `labels=` de transformers.
    """
    return input_ids[..., :-1].contiguous(), labels[..., 1:].contiguous()
