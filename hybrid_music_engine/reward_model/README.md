# reward_model — Punto A del plan

Reemplaza las heurísticas de ranking del `generate-ranked` por un **reward
model aprendido sobre tus features musicales**. Sin tocar el código existente:
lee `data/ranked/<id>/ranking.json` y escribe `ranking_reranked.json` al lado.

## Por qué pairwise (Bradley-Terry) y no regresión escalar

La "calidad musical" es ordinal, no absoluta: nadie puede decir "esta canción
vale 0.73". Sí podemos decir "A suena mejor que B". Bradley-Terry modela
exactamente eso: `P(A > B) = sigmoid(r(A) - r(B))`. Es la misma pérdida que
usan los reward models de RLHF (InstructGPT, DPO arranca de aquí).

Ventaja operativa: **arrancar SIN labels humanos**. El módulo `bootstrap`
empareja MIDIs reales del dataset (Jamendo procesado) contra candidatas
generadas — "preferred = real" siempre. El modelo aprende a empujar las
generaciones hacia la distribución de música real. Es un proxy, no un juez
perfecto, pero ya mejora el ranking heurístico de inmediato.

## Estructura

```
reward_model/
├── features.py          Normaliza dict de métricas → vector fijo (sin torch)
├── model.py             MLP + Bradley-Terry loss
├── dataset.py           Carga pares + bootstrap real-vs-generado
├── train.py             Bucle con pairwise loss, early stop, val accuracy
├── score.py             Inferencia: features → score escalar
├── rerank.py            Lee ranking.json → escribe ranking_reranked.json
├── metrics_provider.py  Cómo extraer métricas: API local o pretty_midi
├── api_router.py        Router FastAPI opcional
└── cli.py               python -m instruct_music_engine.reward_model.cli ...
```

## Flujo en tu proyecto (paso a paso)

### 1. Bootstrap inicial

Una sola vez. Construye un dataset de pares automáticos.

```bash
python -m instruct_music_engine.reward_model.cli bootstrap \
    --real-dir      data/datasets/jamendo/<CATALOG>/processed/<BATCH>/midis \
    --generated-dir data/ranked \
    --output        data/reward/bootstrap.jsonl \
    --max-pairs     2000 \
    --metrics-mode  auto
```

`--metrics-mode auto` intenta primero tu API local (`POST /api/metrics/midi`,
exactamente las mismas métricas del pipeline). Si la API no está corriendo,
cae a un fallback local con `pretty_midi`.

### 2. Entrenar

```bash
python -m instruct_music_engine.reward_model.cli train \
    --pairs       data/reward/bootstrap.jsonl \
    --output-dir  data/models/reward/v1 \
    --epochs 30 --batch-size 64 --lr 1e-3
```

Produce:
- `reward_model.pt` — pesos + config (~50 KB)
- `schema.json` — orden de columnas, medianas, z-score (reproducible)
- `history.json` — curvas de pérdida y accuracy

Al final imprime un JSON con `best_val_acc`. Por encima de **0.60** ya supera
ranking aleatorio; por encima de **0.75** está aprendiendo señal útil.

### 3. Re-rankear lo existente

Una candidata o todas las que ya tienes:

```bash
# Una sola
python -m instruct_music_engine.reward_model.cli rerank \
    --ranking data/ranked/<RANKING_ID>/ranking.json \
    --model   data/models/reward/v1/reward_model.pt \
    --schema  data/models/reward/v1/schema.json \
    --alpha   1.0

# Todas
python -m instruct_music_engine.reward_model.cli rerank-all \
    --root    data/ranked \
    --model   data/models/reward/v1/reward_model.pt \
    --schema  data/models/reward/v1/schema.json
```

`--alpha`: `1.0` = solo reward, `0.0` = solo heurística original, intermedio
= mezcla. Empieza con `1.0`; si en escucha notas que pierde alguna métrica
útil (p. ej. preferencia por candidatas más largas), baja a 0.7.

Cada `ranking.json` produce un `ranking_reranked.json` al lado con:
- candidatas reordenadas por `final_score`
- cada candidata con `reward_score`, `heuristic_score_original`, `final_score`, `rank`, `original_index`
- bloque `rerank.spearman_vs_original` para diagnóstico: si está cerca de
  +1 significa que reward y heurística coinciden (el reward está reforzando lo
  que ya tenías); si está negativo o cerca de 0, el reward está cambiando
  decisiones — ahí es donde aporta valor.

### 4. (Opcional) Integrar en la API

En tu `app.py` principal:

```python
from instruct_music_engine.reward_model.api_router import router as reward_router
app.include_router(reward_router, prefix="/api/reward", tags=["reward"])
```

Tres endpoints nuevos:
- `POST /api/reward/score` — body `{"metrics": {...}}` → `{"reward_score": float}`
- `POST /api/reward/rerank` — re-rankea un `ranking.json`
- `POST /api/reward/rerank-all` — re-rankea todos bajo `data/ranked/`

Variable de entorno opcional: `HYBRID_REWARD_DIR=data/models/reward/v1` para
el path por defecto del modelo.

### 5. Añadir preferencias humanas (cuando las tengas)

Formato JSONL en `data/reward/human_pairs.jsonl`:

```json
{"preferred": "data/ranked/A/candidate-02/generated.mid", "rejected": "data/ranked/A/candidate-05/generated.mid", "source": "human"}
{"preferred": "data/ranked/B/candidate-01/metrics.json",   "rejected": {"tempo_bpm": 200, ...}}
```

Re-entrenar combinando humanas (peso ×3 automático) y bootstrap:

```bash
python -m instruct_music_engine.reward_model.cli train \
    --pairs        data/reward/human_pairs.jsonl \
    --extra-pairs  data/reward/bootstrap.jsonl \
    --output-dir   data/models/reward/v2
```

## Decisiones de diseño relevantes

- **NO sobreescribe nada.** El `ranking.json` original se conserva intacto;
  el reranked es un archivo nuevo al lado.
- **Tolerante a esquemas.** `FeatureSchema` reconoce aliases (`tempo`/`tempo_bpm`,
  `density`/`note_density`), imputa con mediana lo que falte y añade banderas
  `__missing` para que el modelo aprenda a desconfiar de imputaciones.
- **Schema versionado.** El `schema.json` se persiste con el modelo. Si en el
  futuro añades nuevas métricas a `/api/metrics/midi`, el viejo modelo las
  ignora hasta que reentrenes con el schema actualizado — sin crasheos.
- **Z-score consistente.** Las medias/desviaciones se ajustan en entrenamiento
  y se aplican IGUAL en inferencia desde `schema.json`.
- **Caché de scorer en la API.** El router mantiene un singleton por
  `(model_dir, device)` para no recargar pesos en cada request.
- **Diagnóstico vs. heurística.** El reporte de rerank incluye Spearman entre
  reward y heurística original; útil para auditar si el modelo está aportando
  o solo replicando lo que ya tenías.

## Tests

```bash
# Sin torch
python tests/test_reward_features.py

# Con torch — entrena un modelo sintético en segundos y valida end-to-end
python tests/test_reward_model.py
```
