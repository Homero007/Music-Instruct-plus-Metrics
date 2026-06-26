# Integración de `new_metrics` en hybrid_music_engine

## 1. Ubicación del subpaquete

Colócalo **paralelo al reward model**, dentro del paquete:

```text
hybrid_music_engine/
  reward_model/        ← ya existe (re-ranking, ANTES del render)
  new_metrics/         ← NUEVO (gráficos, DESPUÉS del render)
    __init__.py
    pipeline.py
    integration.py
    cli.py
    fad_score.py
    clap_score.py
    kld_metric.py
    evaluate_models.py
    genre_tsne.py
    words_tsne.py
    README.md
```

Los scripts de métricas quedan **sin modificar**: `__init__.py`/`pipeline.py`
añaden su propia carpeta a `sys.path`, así que sus imports internos
(`from fad_score import ...`, `from genre_tsne import ...`) siguen funcionando
dentro del paquete.

## 2. Disparo AUTOMÁTICO tras el render

El render de candidatas vive en `generation/ranked.py`, en
`generate_ranked_candidates`. Ese archivo ya dispara el re-ranking automático del
reward model justo antes de `return summary`. Añade el siguiente bloque
**inmediatamente después** de ese `try/except` y **antes** de `return summary`
(mismo estilo no-fatal):

```python
    # Métricas nuevas automáticas (gráficos), DESPUÉS del render.
    try:
        from hybrid_music_engine.new_metrics.integration import run_after_render
        run_after_render(config, run_id, condition_genre=condition_genre)
    except Exception as exc:
        import logging
        logging.getLogger("hybrid_music_engine").warning(
            f"new_metrics automático omitido o fallido: {exc}"
        )

    return summary
```

`config`, `run_id` y `condition_genre` ya están en alcance en esa función.
Como los renders solo existen cuando `render_best=True`, si no hubo render el
disparo no hace nada (devuelve `None`).

### (Opcional) Jobs de render dedicados

Si también quieres métricas tras `render-midi` / `render-layers`, llama a
`run_after_render(config, run_id)` al final de `run_render_midi_job` /
`run_render_layers_job` en `jobs/workflows.py`, usando el `run_id` con que se
escribió la carpeta `data/renders/<run_id>/`.

## 3. Variables de entorno

```bash
HYBRID_NEW_METRICS_AUTO=1                       # 1 (def) corre auto; 0 desactiva
HYBRID_NEW_METRICS_REAL=data/datasets/jamendo/delivery_jamendo_150  # real p/ FAD (opcional)
HYBRID_T5_DIR=data/encodings_v2/t5_seq          # t-SNE de palabras (opcional)
HYBRID_FAD_EXTRACTOR=mel                         # mel (sin checkpoint) | vggish | clap
```

- Sin `HYBRID_NEW_METRICS_REAL`: FAD se omite y se calcula solo **tempo**
  (reference-free). Con él (y subcarpetas por género en ambos lados), FAD se
  calcula por género.
- **KLD no se dispara**: requiere un clasificador que aún no existe. Cuando lo
  tengas, se añade pasando probabilidades a `run_new_metrics(..., kld_real_probs,
  kld_gen_probs)`.

## 4. Salida

```text
data/new_metrics/<run_id>/
  plots/         ← gráficos (FAD/tempo por género, t-SNE palabras si hay T5)
  audio/         ← evaluacion_modelos.xlsx + caché
  tsne/          ← gráficos + coords del t-SNE de palabras
  report.json    ← resumen + rutas de gráficos
```

Esta etapa **nunca** escribe `ranking.json` ni `ranking_reranked.json`: queda
separada del reward model, como se pidió.

## 5. Uso manual (sin esperar al render)

```bash
python -m hybrid_music_engine.new_metrics.cli \
  --generated data/renders/RUN_ID \
  --real data/datasets/jamendo/delivery_jamendo_150 \
  --t5-dir data/encodings_v2/t5_seq \
  --fad-extractor mel
```
