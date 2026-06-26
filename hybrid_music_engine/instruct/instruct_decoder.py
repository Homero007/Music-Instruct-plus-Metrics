"""
instruct_decoder.py — Esqueleto del decoder autorregresivo estilo Instruct-MusicGen.

Implementa el diagrama de la estrategia:

  [Instrucción] ─> T5 ─> Embeddings de texto (L, 768) ──┐
                                                         │ (cross-attn de texto, LoRA)
                                                         ▼
  [Audio a editar] ─> EnCodec ─> embeddings ──> [ Decoder Autorregresivo ] ─> Tokens nuevos
                                  (cross-attn de audio = AudioFusionModule, LoRA)

Cada bloque del decoder tiene tres sub-capas (pre-norm):

  1. Self-attention CAUSAL sobre los tokens musicales generados (base, congelada).
  2. Cross-attention al TEXTO de T5 (proyecciones Q/K/V envueltas en LoRA).
  3. Fusión de AUDIO original (AudioFusionModule, con gating + LoRA en Q/K/V).
  4. Feed-forward (base, congelado).

Es un esqueleto fiel y EJECUTABLE con tensores aleatorios para validar formas y el
flujo de gradiente. No es un checkpoint de MusicGen entrenado: para producción se
cargarían los pesos reales de MusicGen en self-attn/FFN/text-cross-attn y este
módulo aportaría la ruta de fusión de audio + los adaptadores LoRA.

Requiere: torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .audio_fusion import AudioFusionModule, CrossAttention
except ImportError:  # ejecución directa fuera del paquete
    from audio_fusion import AudioFusionModule, CrossAttention  # type: ignore


class CausalSelfAttention(nn.Module):
    """Self-attention con máscara causal. Q/K/V separados para inyección LoRA."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class InstructDecoderBlock(nn.Module):
    """Un bloque del decoder con las cuatro sub-capas pre-norm descritas arriba."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_audio: int,
        dropout: float = 0.0,
        use_audio_fusion: bool = True,
    ):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)

        self.ln_text = nn.LayerNorm(d_model)
        self.text_cross = CrossAttention(d_model, n_heads, dropout)

        self.use_audio_fusion = use_audio_fusion
        if use_audio_fusion:
            self.audio_fusion = AudioFusionModule(d_model, d_audio, n_heads, dropout)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_memory: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        audio_memory: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) self-attention causal sobre tokens musicales
        x = x + self.self_attn(self.ln_self(x))
        # 2) cross-attention a la instrucción de texto (T5)
        x = x + self.text_cross(self.ln_text(x), text_memory, key_padding_mask=text_mask)
        # 3) fusión del audio original (con gating cero-inicializado)
        if self.use_audio_fusion and audio_memory is not None:
            x = self.audio_fusion(x, audio_memory, audio_mask=audio_mask)
        # 4) feed-forward
        x = x + self.ff(self.ln_ff(x))
        return x


class InstructMusicDecoder(nn.Module):
    """
    Decoder autorregresivo sobre tokens EnCodec, condicionado por texto + audio.

    Modela `n_codebooks` cabezales de salida (un logits por codebook RVQ), igual
    que MusicGen. Aquí, por simplicidad del esqueleto, los codebooks se promedian
    en un único embedding de entrada; en una integración real con MusicGen se
    respetaría el patrón de retardo (delay pattern) entre codebooks.
    """

    def __init__(
        self,
        vocab_size: int = 1024,        # tamaño de cada codebook EnCodec
        n_codebooks: int = 4,          # MusicGen usa 4 (encodec_32khz)
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 8,
        d_ff: int = 4096,
        d_audio: int = 128,            # dim del embedding continuo de EnCodec fuente
        max_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.ModuleList(
            [nn.Embedding(vocab_size, d_model) for _ in range(n_codebooks)]
        )
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.text_in = nn.Linear(768, d_model)   # proyecta T5 (768) a d_model

        self.blocks = nn.ModuleList(
            [
                InstructDecoderBlock(d_model, n_heads, d_ff, d_audio, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)]
        )

    def forward(
        self,
        codes: torch.Tensor,             # (B, n_codebooks, T) tokens EnCodec
        text_hidden: torch.Tensor,       # (B, L, 768) secuencia T5
        text_mask: torch.Tensor | None = None,   # (B, L) bool
        audio_embed: torch.Tensor | None = None, # (B, S, d_audio) audio fuente
        audio_mask: torch.Tensor | None = None,  # (B, S) bool
    ) -> torch.Tensor:
        b, k, t = codes.shape
        assert k == self.n_codebooks, f"esperaba {self.n_codebooks} codebooks, recibí {k}"

        # Embedding: suma de los embeddings de cada codebook + posición
        x = sum(self.token_emb[i](codes[:, i, :]) for i in range(k))   # (B, T, d_model)
        positions = torch.arange(t, device=codes.device)
        x = x + self.pos_emb(positions)[None, :, :]

        text_memory = self.text_in(text_hidden)    # (B, L, d_model)

        for block in self.blocks:
            x = block(
                x,
                text_memory=text_memory,
                text_mask=text_mask,
                audio_memory=audio_embed,
                audio_mask=audio_mask,
            )

        x = self.ln_out(x)
        # Logits por codebook: (B, n_codebooks, T, vocab)
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits
