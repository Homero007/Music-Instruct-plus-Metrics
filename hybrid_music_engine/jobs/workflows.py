from __future__ import annotations

import json
from pathlib import Path

from hybrid_music_engine.audio.loader import import_and_normalize_audio
from hybrid_music_engine.audio.stems import separate_stems
from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.datasets.augmentation import augment_midi_dataset
from hybrid_music_engine.datasets.clip_processor import process_jamendo_clips
from hybrid_music_engine.datasets.jamendo import download_jamendo_catalog, prepare_jamendo_clips
from hybrid_music_engine.embeddings.token_vae import (
    encode_token_vae_embedding,
    encode_token_vae_genre_embeddings,
    train_token_vae,
)
from hybrid_music_engine.embeddings.vae_model import encode_project_embedding, train_feature_vae
from hybrid_music_engine.features.global_features import extract_project_features
from hybrid_music_engine.fusion.latent_blend import blend_embedding_files, blend_weighted_embedding_files
from hybrid_music_engine.generation.ranked import generate_ranked_candidates
from hybrid_music_engine.render.pedalboard_engine import mix_layer_renders, render_midi_audio
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
from hybrid_music_engine.storage.job_store import update_job
from hybrid_music_engine.storage.manifest import load_manifest, project_path, save_manifest
from hybrid_music_engine.transcription.drums_onsets import transcribe_drums_onsets
from hybrid_music_engine.transcription.melodic_basic_pitch import transcribe_melodic_basic_pitch
from hybrid_music_engine.tokens.generative_model import (
    generate_tokens_from_model,
    train_token_markov_model,
)
from hybrid_music_engine.tokens.transformer_model import train_token_transformer_model


def run_import_audio_job(job_id: str, project_id: str, source_path: str) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Importando audio",
        message="Copiando y normalizando audio.",
        progress=0.15,
    )
    manifest = load_manifest(config, project_id)
    source_dir = project_path(config, project_id) / "source"
    payload = import_and_normalize_audio(Path(source_path), source_dir, config)
    update_job(
        config,
        job_id,
        status="running",
        stage="Actualizando proyecto",
        message="Guardando análisis inicial.",
        progress=0.85,
    )
    manifest.source = payload
    manifest.status = "audio_imported"
    save_manifest(config, manifest)
    result = {"project_id": project_id, "source": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Audio listo",
        message="Audio importado y normalizado.",
        progress=1.0,
        result=result,
    )
    return result


def run_separate_stems_job(
    job_id: str,
    project_id: str,
    audio_path: str | None = None,
    model_name: str = "htdemucs",
    device: str = "auto",
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando separación",
        message="Localizando audio normalizado.",
        progress=0.1,
    )
    manifest = load_manifest(config, project_id)
    if audio_path:
        source_audio = Path(audio_path)
    else:
        normalized = manifest.source.get("normalized")
        if not normalized:
            raise RuntimeError("El proyecto no tiene audio normalizado. Importa audio primero.")
        source_audio = Path(str(normalized))

    update_job(
        config,
        job_id,
        status="running",
        stage="Separando stems",
        message="Ejecutando Demucs. Esta tarea puede tardar varios minutos.",
        progress=0.25,
    )
    stems_dir = project_path(config, project_id) / "stems"
    payload = separate_stems(
        source_audio,
        stems_dir,
        model_name=model_name,
        device=device,
    )
    update_job(
        config,
        job_id,
        status="running",
        stage="Actualizando proyecto",
        message="Guardando rutas de stems.",
        progress=0.9,
    )
    manifest.stems = payload
    manifest.status = "stems_separated"
    save_manifest(config, manifest)
    result = {"project_id": project_id, "stems": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Stems listos",
        message="Separación completada.",
        progress=1.0,
        result=result,
    )
    return result


def run_transcribe_melodic_job(
    job_id: str,
    project_id: str,
    stems: list[str] | None = None,
    audio_path: str | None = None,
    minimum_note_length: float | None = None,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando transcripción",
        message="Localizando stems melódicos.",
        progress=0.1,
    )
    manifest = load_manifest(config, project_id)
    project_dir = project_path(config, project_id)
    midi_dir = project_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, Path] = {}
    if audio_path:
        sources["melody"] = Path(audio_path)
    else:
        available = manifest.stems.get("files", {})
        selected = stems or ["bass", "vocals", "other"]
        for stem_name in selected:
            path = available.get(stem_name)
            if path:
                sources[stem_name] = Path(str(path))
        if not sources:
            raise RuntimeError(
                "No hay stems melódicos disponibles. Ejecuta separate-stems o indica audio_path."
            )

    outputs: dict[str, dict] = {}
    total = max(len(sources), 1)
    for index, (name, source) in enumerate(sources.items(), start=1):
        update_job(
            config,
            job_id,
            status="running",
            stage=f"Transcribiendo {name}",
            message="Ejecutando Basic Pitch.",
            progress=0.15 + (0.75 * (index - 1) / total),
        )
        outputs[name] = transcribe_melodic_basic_pitch(
            source,
            midi_dir / f"{name}.mid",
            minimum_note_length=minimum_note_length,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )

    manifest.midis = {
        **manifest.midis,
        "melodic": outputs,
    }
    manifest.status = "melodic_midis_transcribed"
    save_manifest(config, manifest)
    result = {"project_id": project_id, "midis": outputs}
    update_job(
        config,
        job_id,
        status="completed",
        stage="MIDI melódico listo",
        message="Transcripción melódica completada.",
        progress=1.0,
        result=result,
    )
    return result


def run_transcribe_drums_job(
    job_id: str,
    project_id: str,
    audio_path: str | None = None,
    bpm: float | None = None,
    onset_delta: float = 0.07,
    onset_wait: float = 0.03,
    note_length: float = 0.08,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando batería",
        message="Localizando stem de drums.",
        progress=0.1,
    )
    manifest = load_manifest(config, project_id)
    project_dir = project_path(config, project_id)
    midi_dir = project_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    if audio_path:
        source_audio = Path(audio_path)
    else:
        drums = manifest.stems.get("files", {}).get("drums")
        if not drums:
            raise RuntimeError(
                "No hay stem de drums disponible. Ejecuta separate-stems o indica audio_path."
            )
        source_audio = Path(str(drums))

    update_job(
        config,
        job_id,
        status="running",
        stage="Detectando golpes",
        message="Analizando transitorios y clasificando kick, snare y hat.",
        progress=0.35,
    )
    payload = transcribe_drums_onsets(
        source_audio,
        midi_dir / "drums.mid",
        bpm=bpm,
        onset_delta=onset_delta,
        onset_wait=onset_wait,
        note_length=note_length,
    )

    update_job(
        config,
        job_id,
        status="running",
        stage="Actualizando proyecto",
        message="Guardando MIDI percusivo.",
        progress=0.9,
    )
    manifest.midis = {
        **manifest.midis,
        "drums": payload,
    }
    manifest.status = "drums_midi_transcribed"
    save_manifest(config, manifest)
    result = {"project_id": project_id, "midi": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="MIDI de batería listo",
        message="Transcripción percusiva completada.",
        progress=1.0,
        result=result,
    )
    return result


def run_extract_features_job(
    job_id: str,
    project_id: str,
    include_audio: bool = True,
    include_midis: bool = True,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando features",
        message="Leyendo audio y MIDI disponibles del proyecto.",
        progress=0.1,
    )
    manifest = load_manifest(config, project_id)
    midi_dir = project_path(config, project_id) / "midis"
    if include_audio and not manifest.source.get("normalized"):
        raise RuntimeError("No hay audio normalizado. Ejecuta import-audio primero.")
    if include_midis and not manifest.midis and not list(midi_dir.glob("*.mid")):
        raise RuntimeError("No hay MIDIs disponibles. Ejecuta transcribe-melodic o transcribe-drums.")

    update_job(
        config,
        job_id,
        status="running",
        stage="Calculando métricas",
        message="Extrayendo features de audio, ritmo y MIDI por capas.",
        progress=0.45,
    )
    payload = extract_project_features(
        manifest,
        config,
        include_audio=include_audio,
        include_midis=include_midis,
    )

    update_job(
        config,
        job_id,
        status="running",
        stage="Guardando features",
        message="Actualizando manifiesto del proyecto.",
        progress=0.9,
    )
    manifest.features = payload
    manifest.status = "features_extracted"
    save_manifest(config, manifest)
    result = {"project_id": project_id, "features": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Features listas",
        message="Extracción de features completada.",
        progress=1.0,
        result=result,
    )
    return result


def run_train_vae_job(
    job_id: str,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando VAE",
        message="Cargando features de proyectos disponibles.",
        progress=0.1,
    )
    update_job(
        config,
        job_id,
        status="running",
        stage="Entrenando VAE",
        message="Optimizando espacio latente sobre features musicales.",
        progress=0.35,
    )
    payload = train_feature_vae(
        config,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        seed=seed,
    )
    result = {"vae": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="VAE entrenado",
        message="Modelo de embeddings guardado.",
        progress=1.0,
        result=result,
    )
    return result


def run_encode_project_job(
    job_id: str,
    project_id: str,
    model_path: str | None = None,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando embedding",
        message="Cargando features y modelo VAE.",
        progress=0.2,
    )
    manifest = load_manifest(config, project_id)
    payload = encode_project_embedding(
        manifest,
        config,
        model_path=Path(model_path) if model_path else None,
    )
    manifest.embeddings = {
        **manifest.embeddings,
        "feature_vae": payload,
    }
    manifest.status = "embedding_encoded"
    save_manifest(config, manifest)
    result = {"project_id": project_id, "embedding": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Embedding listo",
        message="Vector latente guardado en el proyecto.",
        progress=1.0,
        result=result,
    )
    return result


def run_augment_midi_job(
    job_id: str,
    catalog_path: str | None = None,
    source_dir: str | None = None,
    output_name: str = "augmented_midis",
    transpose_steps: list[int] | None = None,
    velocity_jitter: int = 8,
    timing_jitter_ticks: int = 12,
    quantize_step_ticks: int | None = None,
    tempo_scale: float = 1.0,
    seed: int = 42,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Augmentando MIDI",
        message="Creando variantes musicales reproducibles.",
        progress=0.25,
    )
    payload = augment_midi_dataset(
        config,
        catalog_path=Path(catalog_path) if catalog_path else None,
        source_dir=Path(source_dir) if source_dir else None,
        output_name=output_name,
        transpose_steps=transpose_steps,
        velocity_jitter=velocity_jitter,
        timing_jitter_ticks=timing_jitter_ticks,
        quantize_step_ticks=quantize_step_ticks,
        tempo_scale=tempo_scale,
        seed=seed,
    )
    result = {"augmentation": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Augmentación lista",
        message="Dataset aumentado y catálogo generados.",
        progress=1.0,
        result=result,
    )
    return result


def run_train_token_vae_job(
    job_id: str,
    token_manifest_path: str,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Entrenando Token-VAE",
        message="Aprendiendo espacio latente desde secuencias tokenizadas.",
        progress=0.35,
    )
    payload = train_token_vae(
        config,
        token_manifest_path=Path(token_manifest_path),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        seed=seed,
    )
    result = {"token_vae": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Token-VAE entrenado",
        message="Modelo latente de tokens guardado.",
        progress=1.0,
        result=result,
    )
    return result


def run_encode_token_vae_job(
    job_id: str,
    token_source_path: str,
    model_path: str | None = None,
    output_name: str = "token_embedding",
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Codificando Token-VAE",
        message="Convirtiendo tokens en embedding latente.",
        progress=0.45,
    )
    payload = encode_token_vae_embedding(
        config,
        token_source_path=Path(token_source_path),
        model_path=Path(model_path) if model_path else None,
        output_name=output_name,
    )
    result = {"token_embedding": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Embedding Token-VAE listo",
        message="Embedding guardado para condicionar generación.",
        progress=1.0,
        result=result,
    )
    return result


def run_download_jamendo_job(
    job_id: str,
    genre_tags: dict[str, list[str]] | None = None,
    catalog_name: str = "mtg_jamendo",
    tracks_per_page: int = 200,
    max_tracks_per_genre: int | None = 500,
    download_audio: bool = True,
    client_id: str = "b6747d04",
    source: str = "mtg-cdn",
    concurrent_downloads: int = 16,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando Jamendo",
        message="Configurando géneros, tags y carpeta destino.",
        progress=0.05,
    )
    update_job(
        config,
        job_id,
        status="running",
        stage="Descargando Jamendo",
        message="Descargando metadata y audio MTG-Jamendo. Puede tardar bastante.",
        progress=0.25,
    )
    catalog = download_jamendo_catalog(
        config,
        genre_tags=genre_tags,
        client_id=client_id,
        catalog_name=catalog_name,
        tracks_per_page=tracks_per_page,
        max_tracks_per_genre=max_tracks_per_genre,
        download_audio=download_audio,
        source=source,
        concurrent_downloads=concurrent_downloads,
    )
    result = {"catalog": catalog}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Jamendo listo",
        message="Catálogo y descargas Jamendo finalizadas.",
        progress=1.0,
        result=result,
    )
    return result


def run_prepare_jamendo_clips_job(
    job_id: str,
    catalog_path: str,
    clip_duration_seconds: float = 20.0,
    hop_duration_seconds: float | None = None,
    max_clips_per_track: int | None = None,
    min_clip_seconds: float = 5.0,
    sample_rate: int | None = None,
    mono: bool = True,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando clips",
        message="Leyendo catálogo Jamendo y verificando audios descargados.",
        progress=0.1,
    )
    update_job(
        config,
        job_id,
        status="running",
        stage="Cortando audio",
        message="Generando clips WAV por género.",
        progress=0.35,
    )
    catalog = prepare_jamendo_clips(
        config,
        catalog_path=Path(catalog_path),
        clip_duration_seconds=clip_duration_seconds,
        hop_duration_seconds=hop_duration_seconds,
        max_clips_per_track=max_clips_per_track,
        min_clip_seconds=min_clip_seconds,
        sample_rate=sample_rate,
        mono=mono,
    )
    result = {"clip_catalog": catalog}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Clips listos",
        message="Catálogo de clips Jamendo generado.",
        progress=1.0,
        result=result,
    )
    return result


def run_encode_genre_embeddings_job(
    job_id: str,
    token_manifest_path: str,
    model_path: str | None = None,
    output_name: str = "genre_embeddings",
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Codificando géneros",
        message="Creando un embedding Token-VAE por género detectado en el manifest.",
        progress=0.45,
    )
    payload = encode_token_vae_genre_embeddings(
        config,
        token_manifest_path=Path(token_manifest_path),
        model_path=Path(model_path) if model_path else None,
        output_name=output_name,
    )
    result = {"genre_embeddings": payload}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Embeddings por género listos",
        message="Ya se pueden fusionar géneros con pesos configurables.",
        progress=1.0,
        result=result,
    )
    return result


def run_process_jamendo_clips_job(
    job_id: str,
    clips_catalog_path: str,
    max_clips: int | None = None,
    run_stems: bool = False,
    run_melodic: bool = True,
    run_drums: bool = True,
    run_features: bool = True,
    run_tokens: bool = True,
    continue_on_error: bool = True,
    processing_mode: str = "quick",
    midi_cleanup: bool = False,
    quantize_grid: str = "1/16",
    strict_demucs: bool = False,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Procesando clips",
        message="Convirtiendo clips Jamendo en material MIDI/token entrenable.",
        progress=0.2,
    )
    batch = process_jamendo_clips(
        config,
        clips_catalog_path=Path(clips_catalog_path),
        max_clips=max_clips,
        run_stems=run_stems,
        run_melodic=run_melodic,
        run_drums=run_drums,
        run_features=run_features,
        run_tokens=run_tokens,
        continue_on_error=continue_on_error,
        processing_mode=processing_mode,
        midi_cleanup=midi_cleanup,
        quantize_grid=quantize_grid,
        strict_demucs=strict_demucs,
    )
    result = {"batch": batch}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Batch listo",
        message="Clips procesados y manifest generado.",
        progress=1.0,
        result=result,
    )
    return result


def run_train_token_model_job(
    job_id: str,
    token_manifest_path: str,
    model_name: str = "token_markov",
    order: int = 2,
    model_type: str = "markov",
    sequence_length: int = 128,
    epochs: int = 8,
    batch_size: int = 16,
    embedding_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Entrenando tokens",
        message=f"Construyendo modelo generativo {model_type} sobre tokens MIDI.",
        progress=0.4,
    )
    if model_type == "transformer":
        model = train_token_transformer_model(
            config,
            token_manifest_path=Path(token_manifest_path),
            model_name=model_name,
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    elif model_type == "markov":
        model = train_token_markov_model(
            config,
            token_manifest_path=Path(token_manifest_path),
            model_name=model_name,
            order=order,
        )
    else:
        raise RuntimeError("model_type debe ser markov o transformer.")
    result = {"model": model}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Modelo token listo",
        message="Modelo generativo guardado.",
        progress=1.0,
        result=result,
    )
    return result


def run_generate_tokens_job(
    job_id: str,
    model_path: str,
    duration_seconds: float,
    output_name: str = "generated",
    seed: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    condition_genre: str | None = None,
    feature_tokens: list[str] | None = None,
    embedding_path: str | None = None,
    token_vae_embedding_path: str | None = None,
    export_layers: bool = True,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Generando MIDI",
        message="Sampleando tokens y convirtiendo a MIDI con duración objetivo.",
        progress=0.4,
    )
    generation = generate_tokens_from_model(
        config,
        model_path=Path(model_path),
        duration_seconds=duration_seconds,
        output_name=output_name,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        condition_genre=condition_genre,
        feature_tokens=feature_tokens,
        embedding_path=Path(token_vae_embedding_path or embedding_path)
        if (token_vae_embedding_path or embedding_path)
        else None,
        export_layers=export_layers,
    )
    result = {"generation": generation}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Generación lista",
        message="Tokens y MIDI generados.",
        progress=1.0,
        result=result,
    )
    return result


def run_generate_ranked_job(
    job_id: str,
    model_path: str,
    duration_seconds: float,
    output_name: str = "ranked_generation",
    candidates: int = 6,
    seed: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    condition_genre: str | None = None,
    feature_tokens: list[str] | None = None,
    embedding_path: str | None = None,
    token_vae_embedding_path: str | None = None,
    export_layers: bool = True,
    render_best: bool = False,
    render_engine: str = "auto",
    soundfont_path: str | None = None,
    export_mp3: bool = False,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Generando candidatos",
        message=f"Generando y rankeando {candidates} versiones.",
        progress=0.2,
    )
    ranking = generate_ranked_candidates(
        config,
        model_path=Path(model_path),
        duration_seconds=duration_seconds,
        output_name=output_name,
        candidates=candidates,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        condition_genre=condition_genre,
        feature_tokens=feature_tokens,
        embedding_path=Path(token_vae_embedding_path or embedding_path)
        if (token_vae_embedding_path or embedding_path)
        else None,
        export_layers=export_layers,
        render_best=render_best,
        render_engine=render_engine,
        soundfont_path=Path(soundfont_path) if soundfont_path else None,
        export_mp3=export_mp3,
    )
    result = {"ranking": ranking}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Ranking listo",
        message="Candidatos generados, medidos y ordenados.",
        progress=1.0,
        result=result,
    )
    return result


def run_blend_embeddings_job(
    job_id: str,
    embedding_a_path: str,
    embedding_b_path: str,
    alpha: float = 0.5,
    output_name: str = "latent_blend",
) -> dict:
    config = EngineConfig.from_env()
    blend = blend_embedding_files(
        config,
        embedding_a_path=Path(embedding_a_path),
        embedding_b_path=Path(embedding_b_path),
        alpha=alpha,
        output_name=output_name,
    )
    result = {"blend": blend}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Fusión lista",
        message="Embedding híbrido guardado.",
        progress=1.0,
        result=result,
    )
    return result


def run_blend_weighted_embeddings_job(
    job_id: str,
    embeddings: list[dict],
    output_name: str = "genre_fusion",
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Fusionando géneros",
        message="Mezclando embeddings latentes con pesos configurables.",
        progress=0.55,
    )
    blend = blend_weighted_embedding_files(
        config,
        embeddings=embeddings,
        output_name=output_name,
    )
    result = {"blend": blend}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Fusión lista",
        message="Embedding híbrido guardado para generación.",
        progress=1.0,
        result=result,
    )
    return result


def run_compare_fusions_job(
    job_id: str,
    model_path: str,
    fusion_embeddings: list[dict],
    duration_seconds: float = 30,
    output_name: str = "fusion_comparison",
    candidates_per_fusion: int = 3,
    seed: int | None = 42,
    max_tokens: int | None = 1200,
    temperature: float = 0.84,
    top_k: int | None = 56,
    top_p: float | None = 0.92,
    feature_tokens: list[str] | None = None,
    export_layers: bool = True,
    render_best: bool = False,
    render_engine: str = "auto",
    soundfont_path: str | None = None,
    export_mp3: bool = False,
) -> dict:
    config = EngineConfig.from_env()
    if not fusion_embeddings:
        raise RuntimeError("Indica al menos una fusión para comparar.")
    comparison_id = Path(output_name).stem or "fusion_comparison"
    comparison_dir = config.data_dir / "fusion_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    update_job(
        config,
        job_id,
        status="running",
        stage="Comparando fusiones",
        message=f"Probando {len(fusion_embeddings)} embeddings híbridos con la misma configuración.",
        progress=0.1,
    )
    rows: list[dict] = []
    total = len(fusion_embeddings)
    for index, item in enumerate(fusion_embeddings, start=1):
        label = str(item.get("label") or Path(str(item.get("embedding_path") or item.get("path"))).stem)
        embedding = str(item.get("embedding_path") or item.get("path") or "")
        if not embedding:
            raise RuntimeError("Cada fusión debe incluir embedding_path.")
        ranking = generate_ranked_candidates(
            config,
            model_path=Path(model_path),
            duration_seconds=duration_seconds,
            output_name=f"{output_name}_{_safe_workflow_name(label)}",
            candidates=candidates_per_fusion,
            seed=(seed + ((index - 1) * candidates_per_fusion)) if seed is not None else None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            feature_tokens=feature_tokens,
            embedding_path=Path(embedding),
            export_layers=export_layers,
            render_best=render_best,
            render_engine=render_engine,
            soundfont_path=Path(soundfont_path) if soundfont_path else None,
            export_mp3=export_mp3,
        )
        candidate_scores = [float(candidate.get("score", 0.0)) for candidate in ranking.get("candidates", [])]
        rows.append(
            {
                "label": label,
                "embedding_path": str(Path(embedding).expanduser().resolve()),
                "ranking_path": ranking.get("path"),
                "best_candidate_id": ranking.get("best_candidate_id"),
                "best_score": float(ranking.get("best_score", 0.0)),
                "average_score": round(sum(candidate_scores) / len(candidate_scores), 8)
                if candidate_scores
                else 0.0,
                "candidates": len(candidate_scores),
            }
        )
        update_job(
            config,
            job_id,
            status="running",
            stage="Comparando fusiones",
            message=f"Fusión {index}/{total}: {label}",
            progress=0.1 + (0.8 * index / total),
        )
    rows.sort(key=lambda row: (row["best_score"], row["average_score"]), reverse=True)
    summary_path = comparison_dir / f"{_safe_workflow_name(comparison_id)}.json"
    summary = {
        "schema_version": "fusion-comparison-v1",
        "comparison_id": comparison_id,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "duration_seconds": duration_seconds,
        "candidates_per_fusion": candidates_per_fusion,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "feature_tokens": feature_tokens or [],
        "best_fusion": rows[0] if rows else None,
        "results": rows,
        "path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result = {"comparison": summary}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Comparación lista",
        message="Fusiones generadas y ordenadas por score.",
        progress=1.0,
        result=result,
    )
    return result


def _safe_workflow_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.lower())
    return cleaned.strip("_") or "fusion"


def run_render_midi_job(
    job_id: str,
    midi_path: str,
    output_name: str = "preview",
    engine: str = "auto",
    soundfont_path: str | None = None,
    sample_rate: int = 44100,
    export_mp3: bool = False,
    pedalboard_preset: str = "master",
    plugin_paths: list[str] | None = None,
) -> dict:
    config = EngineConfig.from_env()
    output_dir = config.data_dir / "renders" / output_name
    render = render_midi_audio(
        Path(midi_path),
        output_dir,
        config=config,
        output_name=output_name,
        engine=engine,
        soundfont_path=Path(soundfont_path) if soundfont_path else None,
        sample_rate=sample_rate,
        export_mp3=export_mp3,
        pedalboard_preset=pedalboard_preset,
        plugin_paths=[Path(path) for path in plugin_paths or []],
    )
    result = {"render": render}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Render listo",
        message="WAV preview generado.",
        progress=1.0,
        result=result,
    )
    return result


def run_render_layers_job(
    job_id: str,
    generation_path: str,
    output_name: str = "layer_render",
    engine: str = "auto",
    soundfont_path: str | None = None,
    sample_rate: int = 44100,
    export_mp3: bool = False,
    pedalboard_preset: str = "master",
    plugin_paths: list[str] | None = None,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Renderizando capas",
        message="Leyendo generación y preparando MIDIs por capa.",
        progress=0.15,
    )
    payload = json.loads(Path(generation_path).expanduser().read_text(encoding="utf-8"))
    layer_midis = payload.get("layer_midis") or {}
    if not layer_midis:
        tokens = [str(token) for token in payload.get("tokens", [])]
        if not tokens:
            raise RuntimeError("La generación no contiene tokens ni layer_midis.")
        from hybrid_music_engine.tokens.generative_model import tokens_to_layered_midis

        output_root = Path(str(payload.get("path", generation_path))).expanduser().parent / "layers"
        layer_midis = tokens_to_layered_midis(
            tokens,
            output_root,
            duration_seconds=payload.get("duration_seconds_requested"),
        )
    renders: dict[str, dict] = {}
    total = max(len(layer_midis), 1)
    for index, (layer_name, midi_path) in enumerate(layer_midis.items(), start=1):
        update_job(
            config,
            job_id,
            status="running",
            stage=f"Renderizando {layer_name}",
            message="Convirtiendo MIDI de capa a audio.",
            progress=0.15 + (0.75 * (index - 1) / total),
        )
        renders[layer_name] = render_midi_audio(
            Path(str(midi_path)),
            config.data_dir / "renders" / output_name / layer_name,
            config=config,
            output_name=layer_name,
            engine=engine,
            soundfont_path=Path(soundfont_path) if soundfont_path else None,
            sample_rate=sample_rate,
            export_mp3=export_mp3,
            pedalboard_preset=pedalboard_preset,
            plugin_paths=[Path(path) for path in plugin_paths or []],
        )
    mix = None
    try:
        mix = mix_layer_renders(
            renders,
            config.data_dir / "renders" / output_name / "mix",
            config=config,
            output_name="master",
            export_mp3=export_mp3,
        )
    except RuntimeError as exc:
        mix = {"error": str(exc)}
    result = {"generation_path": generation_path, "renders": renders, "mix": mix}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Capas renderizadas",
        message="Render por capas completado.",
        progress=1.0,
        result=result,
    )
    return result


def run_midi_metrics_job(job_id: str, midi_path: str) -> dict:
    config = EngineConfig.from_env()
    metrics = analyze_midi_quality(Path(midi_path))
    result = {"metrics": metrics}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Métricas listas",
        message="Análisis MIDI completado.",
        progress=1.0,
        result=result,
    )
    return result



def run_evaluation_generate_batch_job(
    job_id: str,
    model_path: str,
    distribution: dict[str, int] | None = None,
    real_audio_root: str | None = None,
    duration_seconds: float = 30.0,
    output_name: str = "evaluation_batch",
    seed: int | None = 42,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    export_layers: bool = True,
    render_audio: bool = True,
    render_engine: str = "auto",
    export_mp3: bool = True,
    target_total: int = 100,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Generando lote de evaluación",
        message="Creando canciones por género y guardando audios/MIDI para métricas.",
        progress=0.05,
    )
    from hybrid_music_engine.evaluation.pipeline import generate_evaluation_batch

    result = generate_evaluation_batch(
        config,
        model_path=Path(model_path),
        distribution=distribution,
        real_audio_root=Path(real_audio_root) if real_audio_root else None,
        duration_seconds=duration_seconds,
        output_name=output_name,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        export_layers=export_layers,
        render_audio=render_audio,
        render_engine=render_engine,
        export_mp3=export_mp3,
        target_total=target_total,
    )
    update_job(
        config,
        job_id,
        status="completed",
        stage="Lote generado",
        message="El lote de evaluación quedó listo para calcular métricas.",
        progress=1.0,
        result=result,
    )
    return result


def run_evaluation_run_job(
    job_id: str,
    evaluation_id: str | None = None,
    generated_root: str | None = None,
    real_root: str | None = None,
    prompts_path: str | None = None,
    classifier_path: str | None = None,
    train_classifier_if_missing: bool = True,
    metrics: list[str] | None = None,
    fad_extractor: str = "mel",
    clap_model: str = "clap",
    device: str = "cpu",
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Calculando métricas",
        message="Ejecutando FAD, KLD, tempo, métricas MIDI y CLAP si fue activado.",
        progress=0.1,
    )
    from hybrid_music_engine.evaluation.pipeline import run_evaluation_metrics

    result = run_evaluation_metrics(
        config,
        evaluation_id=evaluation_id,
        generated_root=Path(generated_root) if generated_root else None,
        real_root=Path(real_root) if real_root else None,
        prompts_path=Path(prompts_path) if prompts_path else None,
        classifier_path=Path(classifier_path) if classifier_path else None,
        train_classifier_if_missing=train_classifier_if_missing,
        metrics=metrics,
        fad_extractor=fad_extractor,
        clap_model=clap_model,
        device=device,
    )
    update_job(
        config,
        job_id,
        status="completed",
        stage="Métricas listas",
        message="Reporte de evaluación generado.",
        progress=1.0,
        result=result,
    )
    return result


def run_evaluation_from_results_job(
    job_id: str,
    selections: list[dict],
    genre_selections: dict[str, list[dict]] | None = None,
    target_per_genre: int = 20,
    pairing_strategy: str = "same_genre_round_robin",
    real_audio_root: str | None = None,
    output_name: str = "generated_results",
    metrics: list[str] | None = None,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Preparando evaluación",
        message="Tomando canciones ya generadas/renderizadas para calcular métricas.",
        progress=0.08,
    )
    from hybrid_music_engine.evaluation.pipeline import (
        create_evaluation_from_results,
        run_evaluation_metrics,
    )

    manifest = create_evaluation_from_results(
        config,
        selections=selections,
        genre_selections=genre_selections or {},
        target_per_genre=target_per_genre,
        pairing_strategy=pairing_strategy,
        real_audio_root=Path(real_audio_root) if real_audio_root else None,
        output_name=output_name,
    )
    update_job(
        config,
        job_id,
        status="running",
        stage="Calculando métricas",
        message="Ejecutando FAD, KLD, tempo, métricas MIDI y CLAP si fue activado.",
        progress=0.35,
        result={"manifest": manifest},
    )
    report = run_evaluation_metrics(
        config,
        evaluation_id=str(manifest["evaluation_id"]),
        metrics=metrics or ["fad", "kld", "tempo", "midi"],
        fad_extractor="mel",
        train_classifier_if_missing=True,
    )
    result = {"manifest": manifest, "report": report, "evaluation_id": manifest["evaluation_id"]}
    update_job(
        config,
        job_id,
        status="completed",
        stage="Evaluación lista",
        message="Las métricas quedaron integradas al ciclo de generación.",
        progress=1.0,
        result=result,
    )
    return result


def run_classifier_train_job(
    job_id: str,
    real_audio_root: str,
    labels: list[str] | None = None,
    output_name: str = "audio_classifier",
    max_files_per_class: int | None = None,
    temperature: float = 1.0,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Entrenando clasificador",
        message="Extrayendo características de audio para probabilidades KLD.",
        progress=0.1,
    )
    from hybrid_music_engine.audio_classifier.model import train_audio_classifier

    result = train_audio_classifier(
        config,
        real_audio_root=Path(real_audio_root),
        labels=labels,
        output_name=output_name,
        max_files_per_class=max_files_per_class,
        temperature=temperature,
    )
    update_job(
        config,
        job_id,
        status="completed",
        stage="Clasificador listo",
        message="El clasificador ya puede producir probabilidades para KLD.",
        progress=1.0,
        result=result,
    )
    return result


def run_classifier_predict_job(
    job_id: str,
    model_path: str | None = None,
    audio_paths: list[str] | None = None,
    audio_root: str | None = None,
) -> dict:
    config = EngineConfig.from_env()
    update_job(
        config,
        job_id,
        status="running",
        stage="Clasificando audio",
        message="Calculando probabilidades por género.",
        progress=0.2,
    )
    from hybrid_music_engine.audio_classifier.model import AudioCentroidClassifier, AUDIO_EXTENSIONS

    selected_model = Path(model_path) if model_path else sorted((config.data_dir / "models" / "audio_classifier").glob("*/classifier.json"))[-1]
    classifier = AudioCentroidClassifier.load(selected_model)
    paths = [Path(item) for item in (audio_paths or [])]
    if audio_root:
        root = Path(audio_root)
        paths.extend(sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS))
    if not paths:
        raise RuntimeError("Indica audio_paths o audio_root para clasificar.")
    probabilities = classifier.predict_batch(paths)
    result = {
        "model_path": str(selected_model),
        "labels": classifier.labels,
        "rows": [
            {"path": str(path), "probabilities": probabilities[index].astype(float).tolist()}
            for index, path in enumerate(paths)
        ],
    }
    update_job(
        config,
        job_id,
        status="completed",
        stage="Audio clasificado",
        message="Probabilidades calculadas.",
        progress=1.0,
        result=result,
    )
    return result
