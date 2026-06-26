"""
audio_fusion.py — Módulo de Fusión de Audio (Audio Fusion Module).

Para EDITAR música, el decoder necesita dos condiciones: la instrucción de texto
(vía T5) y el AUDIO ORIGINAL que se va a modificar.

El problema: concatenar el audio original y el nuevo en la misma línea temporal
duplica la longitud de contexto del transformador y satura la memoria. La solución
de Instruct-MusicGen es un módulo de fusión que inyecta las características del
audio original mediante CROSS-ATTENTION dedicada en bloques del decoder. Así la
estructura temporal del audio fuente guía la generación del nuevo flujo de tokens
EnCodec sin alargar la secuencia.

Diseño clave — GATING CERO-INICIALIZADO:
  La salida de la fusión se multiplica por tanh(gate) con gate=0 al inicio. Por
  tanto, al comenzar el fine-tuning el módulo es la IDENTIDAD (no aporta nada) y
  el modelo se comporta exactamente como el MusicGen preentrenado. El gate se
  abre gradualmente conforme LoRA aprende a usar el audio fuente. Esto evita que
  ruido aleatorio del módulo nuevo destruya la síntesis preentrenada (misma idea
  que el zero-conv de ControlNet y el tanh-gating de Flamingo).

Las proyecciones q_proj/k_proj/v_proj se nombran así a propósito para que lora.py
las pueda envolver con inject_lora(..., target_names=("q_proj","k_proj","v_proj")).

Requiere: torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Atención cruzada multi-cabeza con proyecciones separadas (Q/K/V/out).

    Separamos las proyecciones (en vez de un qkv empaquetado) precisamente para
    poder inyectar LoRA en Q, K, V de forma individual.

    query  : (B, Tq, d_model)   — estados del decoder (tokens musicales en curso)
    memory : (B, Tk, d_model)   — condición (texto T5 o audio fuente proyectado)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) debe ser divisible por n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, tq, _ = query.shape
        tk = memory.shape[1]

        q = self.q_proj(query).view(b, tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) con True = posición válida.
            # Construimos una máscara aditiva (B, 1, 1, Tk) con -inf en el padding.
            attn_mask = torch.zeros(b, 1, 1, tk, device=query.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, tq, self.d_model)
        return self.out_proj(out)


class AudioFusionModule(nn.Module):
    """
    Inyecta el audio original en el decoder por cross-attention con gating.

    Flujo:
      audio_embed (B, S, d_audio)  →  proyección a d_model  →  memoria
      decoder_hidden (B, T, d_model)  →  query
      salida = decoder_hidden + tanh(gate) · CrossAttention(query, memoria)

    `d_audio` es la dimensión del embedding continuo de EnCodec del audio fuente
    (p. ej. 128 para encodec_24khz). Se proyecta a la dimensión del decoder.
    """

    def __init__(
        self,
        d_model: int,
        d_audio: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.audio_in = nn.Linear(d_audio, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout=dropout)
        # Gate escalar cero-inicializado → al inicio la fusión es identidad.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        audio_embed: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.audio_in(audio_embed)                 # (B, S, d_model)
        query = self.norm(decoder_hidden)
        fused = self.cross_attn(query, memory, key_padding_mask=audio_mask)
        return decoder_hidden + torch.tanh(self.gate) * fused

    @property
    def openness(self) -> float:
        """Cuánto está 'abierta' la compuerta (0 = identidad, →1 = fusión plena)."""
        return float(torch.tanh(self.gate).item())
