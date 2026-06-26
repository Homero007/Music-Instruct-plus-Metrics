# Sistema de Transformacion Musical

Sistema modular para preparar audio musical, extraer representaciones MIDI/tokens, entrenar modelos generativos, fusionar generos, generar nuevas piezas y evaluarlas contra musica real.

El flujo principal es cerrado:

```text
audio / Jamendo
  -> seleccion de pistas
  -> clips
  -> stems con Demucs
  -> MIDI, features y tokens
  -> Transformer / Markov y Token-VAE
  -> embeddings y fusion de generos
  -> generacion rankeada
  -> render MIDI/WAV/MP3
  -> evaluacion general, por genero y por cancion
  -> descarga de audios, MIDI, tokens y reportes
```

La interfaz web permite ejecutar el ciclo completo de forma guiada. La API FastAPI y la CLI permiten automatizar el mismo proceso.

## Estructura Principal

- `hybrid_music_engine/api/`: backend FastAPI.
- `hybrid_music_engine/jobs/`: jobs locales o Celery/Redis.
- `hybrid_music_engine/datasets/`: descarga, seleccion y preparacion de datos.
- `hybrid_music_engine/audio/`: importacion, normalizacion y separacion de audio.
- `hybrid_music_engine/midi/`: transcripcion, limpieza y capas MIDI.
- `hybrid_music_engine/token_model/`: modelos Markov/Transformer sobre tokens.
- `hybrid_music_engine/token_vae/`: embeddings latentes y codificacion por genero.
- `hybrid_music_engine/generation/`: generacion, ranking y fusions.
- `hybrid_music_engine/render/`: render WAV/MP3 con SoundFont, FluidSynth y Pedalboard.
- `hybrid_music_engine/evaluation/`: FAD, KLD, tempo, MIDI metrics, comparacion por genero y por cancion.
- `hybrid_music_engine/audio_classifier/`: clasificador ligero usado para KLD.
- `hybrid_music_engine/reward_model/`: scoring y re-ranking.
- `frontend/`: interfaz web estatica.
- `data/`: datasets, modelos, generaciones, renders y evaluaciones.
- `assets/soundfonts/default.sf2`: SoundFont esperado por defecto.

## Instalacion Rapida

Consulta [INSTALL_CLEAN.md](./INSTALL_CLEAN.md) para instalacion completa en macOS y Windows.

Instalacion Python completa recomendada:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e ".[audio,ml,metrics,dev]"
```

En Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[audio,ml,metrics,dev]"
```

## Levantar Servicios

Backend local:

```bash
export HYBRID_ENGINE_JOB_BACKEND=local
export HYBRID_ENGINE_REQUIRE_CELERY=0
.venv/bin/python -m uvicorn hybrid_music_engine.api.main:app --reload --host 127.0.0.1 --port 8100
```

Frontend estatico:

```bash
python3 -m http.server 5173 --directory frontend
```

Abrir:

```text
http://127.0.0.1:5173
```

API:

```text
http://127.0.0.1:8100/docs
```

## Flujo Recomendado En Frontend

1. **Seleccionar/importar audio**
   - Selecciona pistas disponibles por genero.
   - La fuente actual esperada para comparacion real es `data/datasets/jamendo/delivery_jamendo_150/audio/<genero>/`.

2. **Preparar clips**
   - Corta audios largos en clips manejables.
   - El resultado alimenta el procesamiento MIDI/tokens.

3. **Procesar clips para entrenamiento**
   - Modo rapido: util para probar el flujo.
   - Modo Token-VAE con Demucs: separa stems y produce capas mas limpias para embeddings/fusion.
   - Produce MIDI, features, tokens y manifests.

4. **Entrenar modelo**
   - Markov: baseline rapido.
   - Transformer: generador principal de secuencias tokenizadas.
   - Token-VAE: embeddings latentes para condicionamiento y fusion.

5. **Generar musica**
   - Generacion normal por genero.
   - Fusion explicita de generos con embeddings ponderados.
   - Ranking de candidatas por score.

6. **Renderizar audio**
   - Siempre se conserva MIDI.
   - WAV usa FluidSynth/SoundFont.
   - MP3 usa FFmpeg.
   - Pedalboard aplica cadena de master/postprocesamiento.

7. **Evaluar resultados generados**
   - Selecciona corridas por genero.
   - Permite mezclar corridas.
   - Calcula metricas generales, por genero y por cancion generada contra original.
   - Incluye reproductores para escuchar generada y original.

8. **Generar lote nuevo para metricas (avanzado)**
   - Crea nuevas canciones por distribucion configurable.
   - Luego calcula el mismo reporte completo: general, por genero y por cancion.

## Metricas

- **Calidad general:** combinacion de calidad MIDI, validez y reward disponible. Mayor suele ser mejor.
- **FAD:** distancia entre audio real y generado. Menor es mejor. El valor global es el principal; el individual es aproximacion por par.
- **KLD:** diferencia entre distribuciones de genero del clasificador. Menor es mejor.
- **CLAP:** similitud texto-audio. Es opcional y pesado; no bloquea FAD/KLD.
- **Validez MIDI:** porcentaje o valor 1/0 de MIDIs validos.
- **Duracion:** segundos generados.
- **Tempo:** BPM aproximado.
- **Diversidad pitch:** variedad de notas usadas.
- **Diversidad ritmica:** variedad temporal de eventos.
- **Densidad:** notas por segundo.
- **Reward:** score interno para ordenar candidatas; no sustituye FAD/KLD.

## Evaluacion

Existen dos caminos:

- **Evaluar resultados generados:** usa musica ya generada/renderizada. Es el camino recomendado para revisar corridas reales.
- **Lote nuevo de evaluacion:** genera un lote nuevo y despues calcula el mismo reporte completo.

Cada evaluacion se guarda en:

```text
data/evaluations/<evaluation_id>/
  manifest.json
  generated/<genre>/
  real/<genre>/
  metrics/
    summary.json
    genre_summary.json
    pair_metrics.json
    fad.json
    kld.json
    tempo.json
    midi.json
  report.json
```

## API Y Postman

- Endpoints documentados: [API_ENDPOINTS.md](./API_ENDPOINTS.md)
- Coleccion Postman: [postman_collection.json](./postman_collection.json)

Variable Postman principal:

```text
baseUrl = http://127.0.0.1:8100
```

## Variables De Entorno

Usa `.env.example` como referencia. Las variables mas importantes son:

- `HYBRID_ENGINE_ROOT`
- `HYBRID_ENGINE_JOB_BACKEND`
- `HYBRID_ENGINE_REQUIRE_CELERY`
- `HYBRID_ENGINE_BROKER_URL`
- `HYBRID_ENGINE_RESULT_BACKEND`
- `HYBRID_SOUNDFONT_PATH`
- `HYBRID_FLUIDSYNTH_BIN`
- `HYBRID_FFMPEG_BIN`
- `HYBRID_REWARD_DIR`
- `HYBRID_NEW_METRICS_REAL`
- `HYBRID_FAD_EXTRACTOR`

## Validacion Basica

```bash
node --check frontend/app.js
python3 - <<'PY'
import ast
from pathlib import Path
for path in Path("hybrid_music_engine").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST OK")
PY
```

## Notas De Entrega

- No incluir `.venv`.
- No incluir caches: `__pycache__`, `.pytest_cache`, `.DS_Store`, `*.pyc`.
- No incluir renders/evaluaciones/generaciones temporales salvo que sean parte de una demostracion.
- Conservar audios base, modelos finales, SoundFont y documentacion.
