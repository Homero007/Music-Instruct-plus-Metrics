# new_metrics — Etapa de métricas nuevas (post-render)

Etapa **independiente** del reward model. El reward model re-rankea candidatas
*antes* del render; esta etapa corre **después de WAV/MP3** y su objetivo es
**obtener gráficos** de evaluación.

```text
... -> generación por capas -> scoring / re-ranking (Reward Model)
    -> WAV/MP3 -> [new_metrics] -> descarga
```

## Qué incluye

- `fad_score.py`, `clap_score.py`, `kld_metric.py`, `evaluate_models.py` — tus
  scripts de métricas, sin modificar.
- `genre_tsne.py` — primitivas t-SNE (librería).
- `words_tsne.py` — t-SNE **solo de palabras** (tokens T5) por género.
- `pipeline.py` — orquestador `run_new_metrics(...)`.
- `cli.py` — interfaz de línea de comandos.

## Salida (subcarpeta propia, separada del reward model)

```text
data/new_metrics/<run_id>/
  plots/                 ← todos los gráficos (.png) consolidados
  audio/                 ← evaluacion_modelos.xlsx + caché de evaluate_models
  tsne/                  ← gráficos y coords del t-SNE de palabras
  report.json            ← resumen numérico + rutas de gráficos
```

## Uso

```bash
python -m new_metrics.cli \
  --generated data/renders/RENDER_ID \
  --real data/datasets/jamendo/delivery_jamendo_150 \
  --t5-dir data/encodings_v2/t5_seq \
  --fad-extractor mel \
  --out data/new_metrics/demo
```

Añadir CLAP (necesita prompts + checkpoint CLAP) y KLD (probabilidades):

```bash
python -m new_metrics.cli --generated ... --real ... \
  --metrics fad clap tempos_std --prompts prompts.csv \
  --kld-real real_probs.npy --kld-gen gen_probs.npy
```

## Notas

- **FAD/CLAP/KLD no se mezclan con el reward model**: esta etapa nunca escribe
  `ranking.json` ni `ranking_reranked.json`.
- **FAD** por género requiere subcarpetas por género en `--generated` y `--real`.
- **CLAP** y los extractores `vggish`/`clap` de FAD requieren descargar
  checkpoints (red). `--fad-extractor mel` no requiere checkpoint.
- **KLD** requiere un clasificador que produzca probabilidades por clase; aquí se
  consume desde archivos `.npy/.csv/.json`. Si aún no hay clasificador, KLD se
  omite.
- **t-SNE** es exploratorio (solo palabras): conserva estructura local; las
  distancias entre clusters lejanos no son interpretables.
```
