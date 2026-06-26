# instruct_music_engine

Carpeta **paralela** al motor original que lo reorienta hacia **edición musical por
instrucciones** estilo *Instruct-MusicGen*, sin tocar tu código existente
(`encode_audio_text.py` y los demás scripts quedan intactos).

Implementa los cuatro pilares que pediste: captions ricos para T5, secuencia T5
completa `(L, 768)`, módulo de fusión de audio por *cross-attention* y LoRA
quirúrgico sobre las proyecciones de atención cruzada.

---

## Mapa: tu especificación → archivo

| Punto de tu estrategia | Implementado en |
|---|---|
| 1. Captions ricos en vez de etiquetas simples | `caption_builder.py` |
| 2. Secuencia T5 completa `(L, 768)`, sin pooling | `text_encoder.py` |
| Edición: T5 entiende comandos de acción | `caption_builder.build_instruction` + `text_encoder` |
| Módulo de fusión de audio (cross-attention) | `audio_fusion.py` |
| LoRA solo en Q/K/V de atención cruzada | `lora.py` |
| Cableado del decoder autorregresivo (tu diagrama) | `instruct_decoder.py` |
| Pipeline/CLI que une todo | `encode_audio_text_v2.py` |

---

## 1. Captions ricos (`caption_builder.py`)

T5 es un modelo de lenguaje: cuanto más denso el texto, más información semántica
capturan sus estados ocultos. En vez de pasarle `"Jazz"`, se construye:

```
Género: Jazz Latino | Instrumentos: Trompeta, Congas, Piano | Tempo: 120 BPM | Estilo: Sincopado, en vivo
```

```python
from instruct_music_engine import TrackMeta, build_caption, build_instruction

build_caption(TrackMeta(genre="Jazz Latino",
                        instruments=["Trompeta", "Congas", "Piano"],
                        tempo_bpm=120, style="Sincopado, en vivo"))

# Comandos de edición canónicos (en inglés, alineados con MusicGen):
build_instruction("remove", "the drums", keep="the bassline")
# -> "Remove the drums but keep the bassline"
```

## 2. Secuencia T5 completa (`text_encoder.py`)

Tu script original hacía *mean-pooling* y devolvía `(768,)`, perdiendo qué token
es "Trompeta", cuál "120 BPM" y cuál el verbo "Remove". Aquí se preserva la
secuencia entera, que es **exactamente** lo que MusicGen consume por
cross-attention (MusicGen nunca hace pooling del texto):

```python
from instruct_music_engine.text_encoder import T5SequenceEncoder

enc = T5SequenceEncoder(device="cpu")          # T5 se carga congelado
out = enc.encode_sequence("Add a piano solo")  # EncodedText
hidden, mask = out.single()                    # (L, 768), (L,)
```

Se conserva `encode_pooled()` por compatibilidad, pero el camino principal es la
secuencia. Cada secuencia se cachea por pista en `.npz` (float16) sin padding.

## 3. Fusión de audio (`audio_fusion.py`)

Para editar, el decoder necesita el **audio original**. En vez de concatenarlo en
la misma línea temporal (lo que duplicaría el contexto y saturaría la memoria),
`AudioFusionModule` lo inyecta por **cross-attention dedicada**: el audio fuente
(embedding continuo de EnCodec) es la *memoria* y los tokens en generación son las
*queries*.

Detalle clave — **gating cero-inicializado**: la salida de la fusión se multiplica
por `tanh(gate)` con `gate=0` al inicio, así el módulo arranca como **identidad** y
el modelo se comporta igual que el MusicGen preentrenado. La compuerta se abre sola
en el primer paso de entrenamiento (su gradiente inicial no es nulo) y deja entrar
la señal del audio gradualmente. Misma idea que el *zero-conv* de ControlNet.

## 4. LoRA quirúrgico (`lora.py`)

Se congela todo (MusicGen + T5) y se inyectan adaptadores de bajo rango `A·B`
**solo en las proyecciones Q/K/V de la atención cruzada** (texto y fusión de
audio). `B` arranca en cero → al inicio el delta es 0, no se degrada nada.

```python
from instruct_music_engine.lora import inject_lora, mark_only_lora_trainable, count_parameters

for block in decoder.blocks:
    inject_lora(block.text_cross,             r=16, alpha=32)  # Q,K,V de texto
    inject_lora(block.audio_fusion.cross_attn, r=16, alpha=32) # Q,K,V de audio
mark_only_lora_trainable(decoder)             # congela base, entrena solo adaptador
print(count_parameters(decoder))
```

Eficiencia real medida en una config tipo **MusicGen-small** (1024 dim, 24 capas):

```
Parametros totales      : 531,491,864
Entrenables (adaptador) :   4,718,616   (LoRA 4,718,592 + 24 gates)
% entrenable            : 0.89 %
```

## 5. Decoder (`instruct_decoder.py`)

Cablea tu diagrama. Cada bloque (pre-norm) tiene: self-attention causal → cross-
attention al texto T5 → fusión de audio → feed-forward. Es un esqueleto **ejecutable**
con tensores aleatorios para validar formas y gradiente; para producción se cargan
los pesos reales de MusicGen en self-attn/FFN/text-cross y este módulo aporta la
ruta de fusión + los LoRA.

```python
from instruct_music_engine.instruct_decoder import InstructMusicDecoder
import torch

dec = InstructMusicDecoder(n_codebooks=4, d_model=1024, n_layers=24)
logits = dec(
    codes=torch.randint(0, 2048, (B, 4, T)),   # tokens EnCodec en generación
    text_hidden=torch.randn(B, L, 768),         # secuencia T5
    audio_embed=torch.randn(B, S, 128),         # audio original (memoria de fusión)
)   # -> (B, 4, T, vocab)
```

---

## Pipeline completo (`encode_audio_text_v2.py`)

Sucesor directo de tu `encode_audio_text.py`:

```bash
# T5 (captions ricos por género) + EnCodec sobre tus segmentos
python encode_audio_text_v2.py --segments data/segments

# Captions ricos por pista desde un manifiesto con metadatos
python encode_audio_text_v2.py --mode t5 --tracks examples_tracks.csv

# Instrucciones de edición para entrenar el editor
python encode_audio_text_v2.py --mode t5 --edits examples_edits.csv
```

Salida en `data/encodings_v2/`:
- `t5_seq/*.npz` — secuencias `(L, 768)` por género, pista e instrucción.
- `encodec/.../*_codes.npy` — tokens RVQ que el decoder predice.
- `encodec/.../*_embed.npy` — embedding continuo = memoria de fusión de audio.

---

## ⚠️ Nota de compatibilidad con MusicGen

Tu pipeline original usa `facebook/encodec_24khz` (8 codebooks). **MusicGen toma
sus tokens acústicos de `facebook/encodec_32khz` (4 codebooks).** Si el objetivo
es enchufar un checkpoint MusicGen real, `audio_encoder.py` ya usa el de 32 kHz por
defecto. Para experimentos propios e independientes, 24 kHz funciona igual; solo
ajusta `--encodec-model` y `n_codebooks` de forma coherente.

---

## Instalación y pruebas

```bash
pip install -e ".[ml,audio]"     # torch, transformers, librosa, etc.

# Pruebas sin dependencias pesadas (captions / instrucciones):
python tests/test_captions.py

# Pruebas de formas, gating, gradiente e inyección LoRA (requieren torch):
python tests/test_torch_shapes.py
```

## Flujo de fine-tuning sugerido

1. Genera pares de entrenamiento: `(audio_original, instrucción, audio_objetivo)`.
2. Codifica con `encode_audio_text_v2.py`: secuencias T5 de las instrucciones +
   codes/embeddings EnCodec de original y objetivo.
3. Carga MusicGen, congélalo, inyecta `AudioFusionModule` + LoRA en las cross-attn.
4. `mark_only_lora_trainable(model)` y entrena solo el adaptador (~1 % de params)
   con *teacher forcing* sobre los codes del audio objetivo.
5. Inferencia: instrucción + audio original → nuevos codes → decodifica con EnCodec.
