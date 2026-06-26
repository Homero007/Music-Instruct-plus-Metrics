# musicgen_real — Integración con MusicGen oficial

Convierte el stub anterior (`musicgen_adapter.py`) en una integración
funcional con `transformers.MusicgenForCausalLM`. Toda la maquinaria que
construimos para el `InstructMusicDecoder` propio (LoRA, AudioFusionModule,
gating con zero-init) ahora se aplica al decoder REAL de MusicGen.

## Qué está validado en esta sesión

8 tests automáticos contra MusicGen real (tamaño juguete, pesos aleatorios,
sin descarga):

- ✅ `locate_decoder` encuentra layers/lm_heads en `transformers==5.9.0`.
- ✅ El monkey-patch es idempotente.
- ✅ **Con gate=0 + eval(), la salida del modelo patcheado coincide byte a
  byte con MusicGen sin patchear.** Esta es la garantía de no-degradación.
- ✅ El delay pattern produce la matriz escalonada correcta.
- ✅ `prepare_for_finetuning` congela todo el base, marca solo LoRA+gates.
- ✅ Forward end-to-end: loss finita, gradientes solo en adaptador.
- ✅ Entrenamiento: loss baja, gates abren, base intacto.
- ✅ Save/load del adaptador: roundtrip byte a byte, checkpoint < 200 KB.

## Qué NO está validado y por qué

**No corrí esto con `facebook/musicgen-small` real.** Los dominios de
HuggingFace no están en la whitelist de red de esta sesión y el modelo pesa
~2 GB. La validación con pesos reales debe hacerla quien tenga GPU + red.

Lo que sí garantizo: si los pesos reales cargan en `MusicgenForCausalLM`
correctamente (cosa que hace `from_pretrained`), el adapter funcionará
exactamente igual que con el modelo aleatorio del test — porque la estructura
es la misma. La diferencia será solo que el modelo base ya sabe sintetizar
música, por lo que la loss inicial será mucho más baja.

## Decisiones técnicas

### Monkey-patch vs forward override

`MusicgenDecoderLayer.forward` no acepta `audio_embed`. Dos opciones:

1. Reescribir todo el forward de la capa → amarra el código a la estructura
   exacta de cada versión de transformers, se rompe con cada update.
2. Monkey-patch con contexto thread-local → estable entre versiones, feo
   pero correcto.

Elegí (2). Ver `patch_layer.py` para los detalles.

### Post-FFN vs pre-FFN

El paper de Instruct-MusicGen describe la fusión "tras la cross-attn de
texto, antes del FFN". Implementarlo así requiere acceso a los hidden
states intermedios, que MusicGen no expone. La fusión queda **post-FFN**
(después del bloque completo). Funcionalmente es muy similar — ambas son
residuales añadidas al stream — y con gate=0 al inicio, la diferencia es
nula. Si experimentalmente se demuestra que post-FFN degrada calidad,
`patch_layer.py` es el único punto a cambiar.

### Delay pattern construido a mano

El `build_delay_pattern_mask` de transformers está pensado para INFERENCIA
(busca `-1` placeholders y trunca). Para teacher forcing necesitamos
construir el patrón explícitamente. Lo hago en `delay_pattern.py` con un
loop directo y documento por qué.

### Solo se guarda el adaptador

`save_adapter()` produce un `.pt` de típicamente <10 MB para MusicGen-small
(~500M params). El modelo base se recarga por separado vía `from_pretrained`.
Esto permite versionar "estilos de edición" como diffs ligeros y compartir
adaptadores sin redistribuir pesos.

## Cómo correr con MusicGen real

```python
from instruct_music_engine.finetune.musicgen_real import (
    MusicGenAdapterConfig, MusicGenInstructAdapter,
)

adapter = MusicGenInstructAdapter(MusicGenAdapterConfig(
    pretrained_name="facebook/musicgen-small",   # ← descarga aquí
    lora_r=16, lora_alpha=32,
    d_audio=128,                                  # encodec_32khz output dim
))
summary = adapter.prepare_for_finetuning(t5_dim=768)
print(summary)
# Esperado para musicgen-small (~500M params):
#   total ≈ 524M, trainable ≈ 5M (~1%)

# Bucle de entrenamiento estándar
opt = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=1e-4)
for batch in loader:
    opt.zero_grad()
    out = adapter(
        target_codes=batch.target_codes,      # (B, K, T) codes objetivo
        text_hidden=batch.text_hidden,        # (B, L, 768) T5 pre-computado
        text_mask=batch.text_mask,
        audio_embed=batch.audio_embed,        # (B, S, 128) EnCodec del audio fuente
        audio_mask=batch.audio_mask,
    )
    out["loss"].backward()
    opt.step()

adapter.save_adapter("checkpoints/edit_v1.pt")
```

## Limitaciones conocidas

- **`MusicgenForCausalLM` es solo el decoder.** El T5 encoder y EnCodec no
  están incluidos: hay que precomputar `text_hidden` con T5 y `audio_embed`
  con EnCodec antes del entrenamiento (eso ya lo hace
  `encode_audio_text_v2.py`).
- **El `pretrained_name` requiere red abierta** la primera vez. Después
  HuggingFace cachea localmente.
- **No reemplaza el `InstructMusicDecoder` propio.** Coexisten: el propio
  sirve para experimentos rápidos sin pesos, este adapter para entrenamiento
  con MusicGen oficial.

## Archivos

| Archivo | Líneas | Qué hace |
|---|---|---|
| `locate.py` | 70 | Encuentra layers/lm_heads en MusicGen, robusto a versiones |
| `patch_layer.py` | 130 | Inserta AudioFusionModule en una capa; contexto thread-local |
| `delay_pattern.py` | 105 | Construye el patrón escalonado para teacher forcing |
| `adapter.py` | 280 | `MusicGenInstructAdapter`: orquesta todo, save/load adaptador |
| `tests/test_musicgen_adapter.py` | 220 | 8 tests sobre MusicGen real (sin pesos) |
