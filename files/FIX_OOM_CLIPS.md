# Fix: MemoryError al cortar clips ("Unable to allocate ... (25785408, 2) float32")

## Causa

En `datasets/jamendo.py`, `prepare_jamendo_clips` carga **toda** la pista en RAM
antes de rebanar:

```python
audio, loaded_sr = _load_clip_audio(source, sample_rate=sr, mono=mono)  # sf.read(source) entero
```

Una pista de ~9 min estéreo a 48 kHz son ~25.8M frames × 2 canales × 4 bytes ≈
**197 MiB en un solo arreglo**. En una máquina con poca RAM libre (o con varios
workers Celery cortando a la vez), la reserva falla. No tiene que ver con
`new_metrics`: es la etapa de dataset.

## Arreglo: cortar en streaming (leer solo la ventana de cada clip)

1. Copia `clip_cutter.py` a `hybrid_music_engine/datasets/clip_cutter.py`.

2. En `datasets/jamendo.py` añade el import (cerca de los demás):

```python
from hybrid_music_engine.datasets.clip_cutter import cut_track_clips
```

3. Dentro del `for track in catalog.get("entries", []):`, **reemplaza** todo el
   bloque que va desde:

```python
        try:
            audio, loaded_sr = _load_clip_audio(source, sample_rate=sr, mono=mono)
        except (OSError, ValueError, RuntimeError) as exc:
            rejected.append({... "reason": f"audio_read_error: {exc}" ...})
            continue
        ...
        while start_sample < total_samples:
            ...
            sf.write(clip_path, ...)
            entries.append({...})
            counts[genre] = counts.get(genre, 0) + 1
            track_clip_count += 1
            start_sample += hop_samples
```

   por esta llamada única:

```python
        genre_dir = output_root / genre
        clip_entries, rejection = cut_track_clips(
            source, genre_dir, track_id,
            genre=genre, track=track,
            target_sr=sr, mono=mono,
            clip_seconds=clip_duration_seconds,
            hop_seconds=hop,
            min_seconds=min_clip_seconds,
            max_clips=max_clips_per_track,
            create_id=create_id,
        )
        if rejection is not None:
            rejected.append(rejection)
            continue
        entries.extend(clip_entries)
        for clip_entry in clip_entries:
            counts[clip_entry["genre"]] = counts.get(clip_entry["genre"], 0) + 1
```

`create_id` ya está importado en `jamendo.py`. `_load_clip_audio` queda sin uso;
puedes borrarlo o dejarlo.

## Qué se conserva

- Mismo esquema de `entries` y mismas razones de rechazo
  (`audio_shorter_than_min_clip`, `audio_read_error`, etc.).
- Mismo `clips_catalog.json`.
- Mismo formato de salida de cada clip (`sf.write(..., subtype="PCM_24")`).
- Si el sample rate del archivo ≠ objetivo, o soundfile no abre el formato
  (algunos MP3), cae a librosa **leyendo por clip** con `offset`/`duration`
  (tampoco carga toda la pista).

## Verificación

`test_clip_cutter.py` mide con `tracemalloc`:

```text
archivo completo ≈ 31.8 MB | pico full-load=32.6 MB | pico streaming=1.8 MB | un clip ≈ 0.88 MB
```

- Pico de memoria del orden de **un clip**, no de la pista completa.
- Clips **sample-exactos** frente al método anterior.
- El recorrido de ventanas coincide con el loop original.

## Mitigación inmediata (mientras aplicas el parche)

- Procesa con `mono=True` (mitad de memoria).
- Reduce la concurrencia de workers (`--concurrency 1` / `--pool=solo`) para no
  cortar varias pistas a la vez.
