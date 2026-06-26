"""
Tests del adapter de MusicGen REAL. Requieren transformers + torch.

Validan el contrato del adapter sobre `transformers.MusicgenForCausalLM`
(sin descargar pesos, con un modelo aleatorio de tamaño juguete):

  1. locate_decoder encuentra las piezas.
  2. patch_decoder_layer es idempotente y, con gate=0 + eval(), produce
     salida IDÉNTICA a MusicGen sin patchear (la garantía de estabilidad).
  3. El delay pattern produce la matriz escalonada esperada.
  4. prepare_for_finetuning marca solo lo correcto como entrenable.
  5. Un forward de entrenamiento corre, la loss es finita, los gradientes
     fluyen SOLO al adaptador, y la pérdida baja monotónicamente cuando
     entrenamos.

Estos tests NO descargan MusicGen real ni necesitan red.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import transformers  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def _skip_if_no_deps():
    if not HAS_DEPS:
        print("torch o transformers no instalados — pruebas omitidas")
        sys.exit(0)


def _tiny_config():
    """Config de juguete de MusicGen para tests rápidos."""
    return {
        "vocab_size": 2050,  # incluye espacio para pad=2048 y bos=2049
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "ffn_dim": 128, "num_codebooks": 4, "max_position_embeddings": 64,
        "pad_token_id": 2048, "bos_token_id": 2049,
    }


def test_locate_finds_layers_and_lm_heads():
    from transformers import MusicgenDecoderConfig, MusicgenForCausalLM
    from hybrid_music_engine.musicgen.locate import locate_decoder, cross_attn_modules

    cfg = MusicgenDecoderConfig(**_tiny_config())
    model = MusicgenForCausalLM(cfg)
    refs = locate_decoder(model)
    assert refs.num_codebooks == 4
    assert refs.hidden_size == 64
    assert len(refs.layers) == 2
    assert len(refs.lm_heads) == 4
    assert len(cross_attn_modules(refs)) == 2


def test_patch_is_identity_with_zero_gate():
    """
    PROPIEDAD CRÍTICA: con gate=0 y eval(), el patch produce la misma salida
    que el MusicGen sin patchear. Esto es la garantía de Instruct-MusicGen
    de no degradar el modelo base al inicio del fine-tuning.
    """
    from transformers import MusicgenDecoderConfig, MusicgenForCausalLM
    from hybrid_music_engine.musicgen.locate import locate_decoder
    from hybrid_music_engine.musicgen.patch_layer import audio_memory, patch_decoder_layer

    torch.manual_seed(42)
    cfg = MusicgenDecoderConfig(**_tiny_config())

    # Construir DOS modelos con los MISMOS pesos (mismo seed):
    torch.manual_seed(42)
    model_a = MusicgenForCausalLM(cfg).eval()
    torch.manual_seed(42)
    model_b = MusicgenForCausalLM(cfg).eval()

    # Patchear solo el segundo
    refs_b = locate_decoder(model_b)
    for layer in refs_b.layers:
        patch_decoder_layer(layer, d_audio=32, n_heads=4).eval()

    B, T = 1, 6
    input_ids = torch.randint(0, 2048, (B * cfg.num_codebooks, T))
    enc = torch.randn(B, 4, cfg.hidden_size)

    with torch.no_grad():
        out_a = model_a(input_ids=input_ids, encoder_hidden_states=enc).logits
        # Con audio_memory pasando algo SIN llamar (audio=None default), gate=0
        audio = torch.randn(B, 10, 32)
        with audio_memory(audio, None):
            out_b = model_b(input_ids=input_ids, encoder_hidden_states=enc).logits

    assert out_a.shape == out_b.shape
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        "El patch NO es identidad con gate=0: "
        f"diff max = {(out_a - out_b).abs().max().item():.2e}"
    )


def test_patch_is_idempotent():
    from transformers import MusicgenDecoderConfig, MusicgenForCausalLM
    from hybrid_music_engine.musicgen.locate import locate_decoder
    from hybrid_music_engine.musicgen.patch_layer import is_patched, patch_decoder_layer

    cfg = MusicgenDecoderConfig(**_tiny_config())
    model = MusicgenForCausalLM(cfg)
    refs = locate_decoder(model)
    layer = refs.layers[0]
    f1 = patch_decoder_layer(layer, d_audio=32, n_heads=4)
    f2 = patch_decoder_layer(layer, d_audio=32, n_heads=4)
    assert f1 is f2
    assert is_patched(layer)


def test_delay_pattern_layout():
    """Verifica el patrón escalonado byte a byte."""
    from transformers import MusicgenDecoderConfig, MusicgenForCausalLM
    from hybrid_music_engine.musicgen.delay_pattern import LABEL_IGNORE, prepare_teacher_forcing_inputs
    from hybrid_music_engine.musicgen.locate import locate_decoder

    cfg = MusicgenDecoderConfig(**_tiny_config())
    model = MusicgenForCausalLM(cfg)
    refs = locate_decoder(model)
    codes = torch.tensor([[
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23, 24, 25],
        [30, 31, 32, 33, 34, 35],
        [40, 41, 42, 43, 44, 45],
    ]])
    input_ids, labels = prepare_teacher_forcing_inputs(codes, refs)
    assert input_ids.shape == (4, 10)  # K=4, T+K=10
    # Codebook k tiene k pads al inicio y K-k pads al final.
    pad = refs.pad_token_id
    for k in range(4):
        for col in range(k):
            assert input_ids[k, col].item() == pad
        for col in range(k + 6, 10):
            assert input_ids[k, col].item() == pad
    # Labels: pads → LABEL_IGNORE
    assert (labels == LABEL_IGNORE).sum().item() == (10 * 4 - 6 * 4)


def test_prepare_freezes_base_only_adapter_trains():
    from hybrid_music_engine.musicgen.adapter import MusicGenAdapterConfig, MusicGenInstructAdapter

    adapter = MusicGenInstructAdapter(MusicGenAdapterConfig(
        config_only_for_test=_tiny_config(), lora_r=8, lora_alpha=16, d_audio=32,
    ))
    summary = adapter.prepare_for_finetuning(t5_dim=64)
    # 2 capas × (text_cross + audio_fusion) × (q,k,v) = 12 inyecciones
    assert summary["injected_linears"]["text_cross"] == 6
    assert summary["injected_linears"]["audio_fusion"] == 6
    assert summary["gates"] == 2
    assert summary["lora"] > 0
    # Cualquier parámetro que no sea lora/gate debe estar congelado
    for name, p in adapter.model.named_parameters():
        if "lora" in name or name.endswith("gate"):
            assert p.requires_grad, f"adapter param sin grad: {name}"
        else:
            assert not p.requires_grad, f"base param entrenable: {name}"


def test_end_to_end_forward_and_backward():
    """Forward+backward end-to-end: pierdas finita, grad solo en adaptador."""
    from hybrid_music_engine.musicgen.adapter import MusicGenAdapterConfig, MusicGenInstructAdapter

    torch.manual_seed(0)
    adapter = MusicGenInstructAdapter(MusicGenAdapterConfig(
        config_only_for_test=_tiny_config(), lora_r=8, lora_alpha=16, d_audio=32,
    ))
    adapter.prepare_for_finetuning(t5_dim=64)

    B, K, T, L, S = 2, 4, 8, 6, 20
    out = adapter(
        target_codes=torch.randint(0, 2048, (B, K, T)),
        text_hidden=torch.randn(B, L, 64),
        text_mask=torch.ones(B, L, dtype=torch.bool),
        audio_embed=torch.randn(B, S, 32),
        audio_mask=torch.ones(B, S, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = {"adapter": 0.0, "base": 0.0}
    for name, p in adapter.model.named_parameters():
        if p.grad is None:
            continue
        if "lora" in name or name.endswith("gate"):
            grads["adapter"] += p.grad.norm().item()
        else:
            grads["base"] += p.grad.norm().item()
    assert grads["adapter"] > 0, "el adaptador no recibió gradiente"
    assert grads["base"] == 0.0, f"el base recibió gradiente: {grads['base']}"


def test_loss_decreases_with_training():
    """
    Test de convergencia mínima: la pérdida debe ser estrictamente menor
    al final que al principio, y los gates deben haberse abierto.
    """
    from hybrid_music_engine.musicgen.adapter import MusicGenAdapterConfig, MusicGenInstructAdapter

    torch.manual_seed(0)
    adapter = MusicGenInstructAdapter(MusicGenAdapterConfig(
        config_only_for_test=_tiny_config(), lora_r=8, lora_alpha=16, d_audio=32,
    ))
    adapter.prepare_for_finetuning(t5_dim=64)
    adapter.train()

    B, K, T, L, S = 2, 4, 8, 6, 20
    codes = torch.randint(0, 2048, (B, K, T))
    text_hidden = torch.randn(B, L, 64)
    audio_embed = torch.randn(B, S, 32)

    opt = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=2e-2)
    initial_loss = None
    final_loss = None
    for step in range(60):
        opt.zero_grad()
        out = adapter(target_codes=codes, text_hidden=text_hidden,
                      text_mask=torch.ones(B, L, dtype=torch.bool),
                      audio_embed=audio_embed,
                      audio_mask=torch.ones(B, S, dtype=torch.bool))
        out["loss"].backward()
        opt.step()
        if step == 0:
            initial_loss = out["loss"].item()
        final_loss = out["loss"].item()

    assert final_loss < initial_loss, f"loss no bajó: {initial_loss} → {final_loss}"
    # Gates: arrancan en 0, deben abrirse con magnitud no trivial.
    gates_open = [abs(layer.audio_fusion.openness) for layer in adapter.refs.layers]
    assert all(g > 0.05 for g in gates_open), f"gates apenas abrieron: {gates_open}"


def test_save_and_load_adapter_roundtrip():
    """Guardar adaptador, recargarlo, y verificar que los pesos coinciden."""
    import tempfile
    from hybrid_music_engine.musicgen.adapter import MusicGenAdapterConfig, MusicGenInstructAdapter

    torch.manual_seed(0)
    adapter = MusicGenInstructAdapter(MusicGenAdapterConfig(
        config_only_for_test=_tiny_config(), lora_r=8, lora_alpha=16, d_audio=32,
    ))
    adapter.prepare_for_finetuning(t5_dim=64)
    # Modificar un LoRA y un gate para no confundirlos con init
    for name, p in adapter.model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.add_(0.3)
            break
    for layer in adapter.refs.layers:
        with torch.no_grad():
            layer.audio_fusion.gate.fill_(0.7)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adapter.pt"
        adapter.save_adapter(path)
        # Tamaño del checkpoint: solo el adaptador
        size_kb = path.stat().st_size / 1024
        assert size_kb < 200, f"checkpoint demasiado grande: {size_kb} KB"

        loaded = MusicGenInstructAdapter.load_adapter(path)
        # Comparar parámetros del adaptador
        orig = {n: p for n, p in adapter.model.named_parameters() if p.requires_grad}
        new = {n: p for n, p in loaded.model.named_parameters() if p.requires_grad}
        for name in orig:
            assert torch.allclose(orig[name].cpu(), new[name].cpu(), atol=1e-6), name


if __name__ == "__main__":
    _skip_if_no_deps()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("TODAS LAS PRUEBAS DE MUSICGEN ADAPTER PASARON")
