# API Endpoints

Base local:

```text
http://127.0.0.1:8100
```

La API usa JSON para entrada/salida. Los procesos pesados devuelven un `job_id`; consulta el estado en `GET /api/jobs/{job_id}`.

Respuesta comun de job:

```json
{
  "job_id": "20260609-job-example",
  "mode": "local-thread"
}
```

Errores comunes:

- `400`: entrada valida, pero no ejecutable: archivo faltante, SoundFont faltante, sin audio renderizado, sin originales por genero.
- `404`: recurso inexistente: proyecto, job, evaluacion o archivo.
- `422`: JSON no cumple el esquema.
- `500`: error interno o metadata corrupta.

Formato de error:

```json
{ "detail": "Mensaje legible" }
```

## Sistema Y Recursos

### GET `/api/health`

Verifica backend, rutas, audio y configuracion.

Salida incluye:

```json
{
  "ok": true,
  "project_root": "...",
  "sample_rate": 48000,
  "channels": 2,
  "job_backend": "local",
  "require_celery": false,
  "audio": {
    "soundfont": true,
    "fluidsynth": true,
    "ffmpeg": true,
    "ready": true
  }
}
```

### GET `/api/resources`

Lista manifests, modelos, generaciones, rankings, renders, blends, Token-VAE y catalogos.

### GET `/api/presets`

Lista presets de entrenamiento/generacion/render.

## Jobs

### GET `/api/jobs`

Lista jobs registrados.

### GET `/api/jobs/{job_id}`

Consulta estado, progreso, eventos y resultado.

## Proyectos

### POST `/api/projects`

```json
{ "name": "mi_proyecto" }
```

### GET `/api/projects`

Lista proyectos.

### GET `/api/projects/{project_id}`

Devuelve manifest del proyecto.

## Pipeline Por Proyecto

### POST `/api/jobs/import-audio`

```json
{
  "project_id": "PROJECT_ID",
  "source_path": "/ruta/audio.wav"
}
```

### POST `/api/jobs/separate-stems`

```json
{
  "project_id": "PROJECT_ID",
  "audio_path": null,
  "model_name": "htdemucs",
  "device": "auto"
}
```

### POST `/api/jobs/transcribe-melodic`

```json
{
  "project_id": "PROJECT_ID",
  "stems": ["bass", "vocals", "other"],
  "audio_path": null,
  "minimum_note_length": null,
  "onset_threshold": null,
  "frame_threshold": null
}
```

### POST `/api/jobs/transcribe-drums`

```json
{
  "project_id": "PROJECT_ID",
  "audio_path": null,
  "bpm": null,
  "onset_delta": 0.07,
  "onset_wait": 0.03,
  "note_length": 0.08
}
```

### POST `/api/jobs/extract-features`

```json
{
  "project_id": "PROJECT_ID",
  "include_audio": true,
  "include_midis": true
}
```

## Datasets Y Jamendo

### GET `/api/datasets/jamendo/catalogs`

Lista catalogos Jamendo disponibles.

### POST `/api/jobs/download-jamendo`

```json
{
  "genre_tags": null,
  "catalog_name": "mtg_jamendo",
  "tracks_per_page": 200,
  "max_tracks_per_genre": 500,
  "download_audio": true,
  "client_id": "b6747d04",
  "source": "mtg-cdn",
  "concurrent_downloads": 16
}
```

### POST `/api/datasets/jamendo/select`

```json
{
  "catalog_path": "data/datasets/jamendo/catalog.json",
  "genres": ["classical", "electronic", "reggaeton"],
  "max_tracks_per_genre": 150,
  "output_name": "delivery_jamendo_150"
}
```

### POST `/api/jobs/prepare-jamendo-clips`

```json
{
  "catalog_path": "data/datasets/jamendo/delivery_jamendo_150/catalog.json",
  "clip_duration_seconds": 20,
  "hop_duration_seconds": null,
  "max_clips_per_track": null,
  "min_clip_seconds": 5,
  "sample_rate": null,
  "mono": true
}
```

### POST `/api/jobs/process-jamendo-clips`

```json
{
  "clips_catalog_path": "data/datasets/jamendo/selected/clips_catalog.json",
  "max_clips": null,
  "run_stems": true,
  "run_melodic": true,
  "run_drums": true,
  "run_features": true,
  "run_tokens": true,
  "continue_on_error": true,
  "processing_mode": "token_vae_demucs",
  "midi_cleanup": true,
  "quantize_grid": "1/16",
  "strict_demucs": true
}
```

Errores esperados: Demucs no disponible, catalogo vacio, clips sin audio util, MIDI/tokens vacios.

## Tokens, Modelos Y Embeddings

### GET `/api/tokens/manifests`

Lista manifests de tokens.

### GET `/api/models/tokens`

Lista modelos Markov/Transformer entrenados.

### POST `/api/jobs/train-token-model`

```json
{
  "token_manifest_path": "data/.../tokens_manifest.json",
  "model_name": "token_transformer",
  "order": 2,
  "model_type": "transformer",
  "sequence_length": 128,
  "epochs": 8,
  "batch_size": 16,
  "embedding_dim": 128,
  "num_layers": 3,
  "num_heads": 4
}
```

### POST `/api/jobs/train-token-vae`

```json
{
  "token_manifest_path": "data/.../tokens_manifest.json",
  "latent_dim": 32,
  "hidden_dim": 128,
  "epochs": 80,
  "learning_rate": 0.001,
  "beta": 0.001,
  "seed": 42
}
```

### POST `/api/jobs/encode-token-vae`

```json
{
  "token_source_path": "data/.../tokens_manifest.json",
  "model_path": null,
  "output_name": "token_embedding"
}
```

### POST `/api/jobs/encode-genre-embeddings`

```json
{
  "token_manifest_path": "data/.../tokens_manifest.json",
  "model_path": null,
  "output_name": "genre_embeddings"
}
```

## Generacion, Fusion Y Render

### POST `/api/jobs/generate-tokens`

```json
{
  "model_path": "data/models/tokens/MODEL_ID/model.json",
  "duration_seconds": 30,
  "output_name": "generated",
  "seed": 42,
  "max_tokens": 1200,
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.95,
  "condition_genre": "electronic",
  "feature_tokens": ["genre:electronic"],
  "embedding_path": null,
  "token_vae_embedding_path": null,
  "export_layers": true
}
```

### POST `/api/jobs/generate-ranked`

```json
{
  "model_path": "data/models/tokens/MODEL_ID/model.json",
  "duration_seconds": 30,
  "output_name": "ranked_generation",
  "candidates": 6,
  "seed": 42,
  "max_tokens": 1200,
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.95,
  "condition_genre": "electronic",
  "feature_tokens": ["genre:electronic"],
  "embedding_path": null,
  "token_vae_embedding_path": null,
  "export_layers": true,
  "render_best": true,
  "render_engine": "pedalboard",
  "soundfont_path": null,
  "export_mp3": true
}
```

### POST `/api/jobs/blend-weighted-embeddings`

```json
{
  "embeddings": [
    { "path": "data/embeddings/genre/classical.pt", "weight": 0.5, "label": "classical" },
    { "path": "data/embeddings/genre/electronic.pt", "weight": 0.5, "label": "electronic" }
  ],
  "output_name": "fusion_classical_electronic"
}
```

### POST `/api/jobs/render-midi`

```json
{
  "midi_path": "data/generated/example.mid",
  "output_name": "preview",
  "engine": "pedalboard",
  "soundfont_path": null,
  "sample_rate": 44100,
  "export_mp3": true,
  "pedalboard_preset": "master",
  "plugin_paths": []
}
```

Errores esperados: SoundFont faltante, FluidSynth faltante, FFmpeg faltante para MP3.

### POST `/api/jobs/render-layers`

```json
{
  "generation_path": "data/generated/GENERATION_ID/generation.json",
  "output_name": "layer_render",
  "engine": "pedalboard",
  "soundfont_path": null,
  "sample_rate": 44100,
  "export_mp3": true,
  "pedalboard_preset": "master",
  "plugin_paths": []
}
```

## Evaluacion Y Metricas

### GET `/api/evaluations/availability`

Consulta audios reales disponibles por genero.

Salida:

```json
{
  "real_audio_root": "data/datasets/jamendo/delivery_jamendo_150/audio",
  "counts": { "classical": 150, "electronic": 150, "reggaeton": 150 },
  "recommended_distribution": { "classical": 34, "electronic": 33, "reggaeton": 33 },
  "max_distribution": { "classical": 150, "electronic": 150, "reggaeton": 150 },
  "target_total": 100,
  "total_available": 450
}
```

### GET `/api/evaluations/generated-sources`

Lista corridas/rankings/renders con audio disponible, agrupados por genero y corrida. Se usa para evaluar resultados ya generados.

### POST `/api/jobs/evaluation/from-results`

Crea una evaluacion desde resultados existentes ya renderizados.

```json
{
  "genre_selections": {
    "classical": [
      { "source_type": "ranking", "source_id": "RANKING_ID", "limit": 20 }
    ],
    "electronic": [
      { "source_type": "ranking", "source_id": "OTHER_RANKING_ID", "limit": 20 }
    ]
  },
  "target_per_genre": 20,
  "pairing_strategy": "same_genre_round_robin",
  "real_audio_root": null,
  "output_name": "resultados_generados",
  "metrics": ["fad", "kld", "tempo", "midi"],
  "include_clap": false
}
```

Campos:

- `genre_selections`: matriz genero -> corridas -> cantidad.
- `target_per_genre`: cantidad objetivo usada por la UI.
- `pairing_strategy`: actualmente `same_genre_round_robin`.
- `include_clap`: agrega CLAP si esta instalado/configurado.

Errores esperados:

- no hay WAV/MP3 renderizado;
- no hay originales para el genero;
- clasificador KLD no disponible;
- CLAP no instalado si se activa.

### POST `/api/jobs/evaluation/generate-batch`

Genera un lote nuevo de canciones y lo prepara para evaluacion.

```json
{
  "model_path": "data/models/tokens/MODEL_ID/model.json",
  "distribution": { "classical": 1, "electronic": 1, "reggaeton": 1 },
  "real_audio_root": null,
  "duration_seconds": 30,
  "output_name": "evaluation_batch",
  "seed": 42,
  "max_tokens": 1200,
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.95,
  "export_layers": true,
  "render_audio": true,
  "render_engine": "pedalboard",
  "export_mp3": true,
  "target_total": 3
}
```

El lote nuevo produce la misma estructura de reporte que `from-results`.

### POST `/api/jobs/evaluation/run`

Calcula metricas para una evaluacion existente.

```json
{
  "evaluation_id": "EVALUATION_ID",
  "generated_root": null,
  "real_root": null,
  "prompts_path": null,
  "classifier_path": null,
  "train_classifier_if_missing": true,
  "metrics": ["fad", "kld", "tempo", "midi"],
  "fad_extractor": "mel",
  "clap_model": "clap",
  "device": "cpu"
}
```

### GET `/api/evaluations`

Lista evaluaciones con resumen y URLs de descarga.

### GET `/api/evaluations/{evaluation_id}`

Devuelve manifest y, si existe, reporte.

### GET `/api/evaluations/{evaluation_id}/report`

Devuelve `report.json`.

### GET `/api/evaluations/{evaluation_id}/files`

Lista archivos descargables de la evaluacion.

## Estructura De Reporte

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

Metricas generales:

- calidad general;
- FAD;
- KLD;
- CLAP opcional;
- validez MIDI;
- duracion promedio;
- tempo promedio;
- diversidad pitch;
- diversidad ritmica.

Metricas por cancion:

- calidad general;
- FAD individual aproximado;
- KLD individual;
- validez MIDI;
- duracion;
- tempo;
- diversidad pitch;
- diversidad ritmica;
- similitud;
- diferencia de duracion;
- diferencia de tempo;
- densidad;
- probabilidad de genero;
- reward.

## Clasificador Para KLD

### POST `/api/jobs/classifier/train`

```json
{
  "real_audio_root": "data/datasets/jamendo/delivery_jamendo_150/audio",
  "labels": ["classical", "electronic", "reggaeton"],
  "output_name": "audio_classifier",
  "max_files_per_class": null,
  "temperature": 1.0
}
```

### POST `/api/jobs/classifier/predict`

```json
{
  "model_path": null,
  "audio_paths": ["data/evaluations/EVAL/generated/classical/song.wav"],
  "audio_root": null
}
```

## Archivos

### GET `/api/files?path=<ruta_absoluta>`

Descarga un archivo permitido dentro del proyecto.

### GET `/api/artifacts/{filename}`

Descarga artefactos generados por nombre.

## Variables De Entorno Relevantes

- `HYBRID_ENGINE_ROOT`
- `HYBRID_ENGINE_JOB_BACKEND`
- `HYBRID_ENGINE_REQUIRE_CELERY`
- `HYBRID_ENGINE_BROKER_URL`
- `HYBRID_ENGINE_RESULT_BACKEND`
- `HYBRID_SOUNDFONT_PATH`
- `HYBRID_FLUIDSYNTH_BIN`
- `HYBRID_FFMPEG_BIN`
- `HYBRID_ENGINE_SAMPLE_RATE`
- `HYBRID_ENGINE_CHANNELS`
- `HYBRID_REWARD_DIR`
- `HYBRID_NEW_METRICS_AUTO`
- `HYBRID_NEW_METRICS_REAL`
- `HYBRID_FAD_EXTRACTOR`
