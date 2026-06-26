"""
Pruebas de formas, gating y conteo de parámetros para los módulos torch.

Se SALTAN automáticamente si torch no está instalado.
Ejecutar:  python -m pytest tests/test_torch_shapes.py -q
o directo: python tests/test_torch_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _skip_if_no_torch():
    if not HAS_TORCH:
        print("torch no instalado — pruebas omitidas")
        sys.exit(0)


def test_lora_zero_init_is_identity():
    import torch.nn as nn
    from hybrid_music_engine.instruct.lora import LoRALinear

    base = nn.Linear(32, 64)
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, 10, 32)
    # B=0 al inicio → la salida debe igualar la del lineal base.
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_trainable_only_lora():
    import torch.nn as nn
    from hybrid_music_engine.instruct.lora import LoRALinear, mark_only_lora_trainable, count_parameters

    net = nn.Sequential(LoRALinear(nn.Linear(16, 16), r=4), nn.ReLU())
    mark_only_lora_trainable(net)
    stats = count_parameters(net)
    assert stats["lora"] > 0
    assert stats["trainable"] == stats["lora"]  # solo LoRA entrena
    assert stats["trainable_pct"] < 50          # eficiente


def test_audio_fusion_starts_as_identity():
    from hybrid_music_engine.instruct.audio_fusion import AudioFusionModule

    mod = AudioFusionModule(d_model=64, d_audio=32, n_heads=8)
    h = torch.randn(2, 20, 64)
    a = torch.randn(2, 15, 32)
    out = mod(h, a)
    # gate=0 → tanh(0)=0 → salida == entrada (fusión identidad al inicio)
    assert torch.allclose(out, h, atol=1e-6)
    assert mod.openness == 0.0


def test_cross_attention_shapes_and_mask():
    from hybrid_music_engine.instruct.audio_fusion import CrossAttention

    attn = CrossAttention(d_model=64, n_heads=8)
    q = torch.randn(2, 10, 64)
    mem = torch.randn(2, 7, 64)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[:, 5:] = False
    out = attn(q, mem, key_padding_mask=mask)
    assert out.shape == (2, 10, 64)


def test_decoder_forward_and_lora_injection():
    from hybrid_music_engine.instruct.instruct_decoder import InstructMusicDecoder
    from hybrid_music_engine.instruct.lora import inject_lora, mark_only_lora_trainable, count_parameters

    dec = InstructMusicDecoder(
        vocab_size=128, n_codebooks=4, d_model=64, n_heads=8,
        n_layers=2, d_ff=128, d_audio=32, max_len=64,
    )
    before = count_parameters(dec)["lora"]
    assert before == 0

    # Inyectar LoRA solo en cross-attention de texto y fusión de audio.
    n = 0
    for block in dec.blocks:
        n += inject_lora(block.text_cross, r=8)
        n += inject_lora(block.audio_fusion.cross_attn, r=8)
    assert n == 2 * 3 * 2  # (text + audio) × (q,k,v) × 2 capas

    mark_only_lora_trainable(dec)
    stats = count_parameters(dec)
    assert stats["lora"] > 0
    assert stats["gates"] == 2          # un gate por AudioFusionModule (2 capas)
    assert stats["trainable"] == stats["adapter"] == stats["lora"] + stats["gates"]
    assert stats["trainable_pct"] < 25  # fine-tuning muy eficiente

    # Forward con tensores aleatorios
    b, t, l, s = 2, 16, 12, 20
    codes = torch.randint(0, 128, (b, 4, t))
    text_hidden = torch.randn(b, l, 768)
    text_mask = torch.ones(b, l, dtype=torch.bool)
    audio_embed = torch.randn(b, s, 32)
    logits = dec(codes, text_hidden, text_mask=text_mask, audio_embed=audio_embed)
    assert logits.shape == (b, 4, t, 128)

    # Gradiente: debe fluir a LoRA, no al base congelado.
    logits.sum().backward()
    grads = [p.grad is not None for p in dec.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


if __name__ == "__main__":
    _skip_if_no_torch()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("TODAS LAS PRUEBAS TORCH PASARON")
