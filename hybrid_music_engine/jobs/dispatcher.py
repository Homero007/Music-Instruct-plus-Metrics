from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.storage.job_store import update_job
from hybrid_music_engine.jobs.workflows import (
    run_augment_midi_job,
    run_encode_token_vae_job,
    run_encode_genre_embeddings_job,
    run_extract_features_job,
    run_encode_project_job,
    run_download_jamendo_job,
    run_import_audio_job,
    run_blend_embeddings_job,
    run_blend_weighted_embeddings_job,
    run_compare_fusions_job,
    run_generate_tokens_job,
    run_generate_ranked_job,
    run_midi_metrics_job,
    run_process_jamendo_clips_job,
    run_render_layers_job,
    run_render_midi_job,
    run_prepare_jamendo_clips_job,
    run_separate_stems_job,
    run_train_vae_job,
    run_train_token_vae_job,
    run_train_token_model_job,
    run_transcribe_drums_job,
    run_transcribe_melodic_job,
)


def _use_celery() -> bool:
    config = EngineConfig.from_env()
    return (
        config.require_celery
        or config.job_backend.lower() == "celery"
        or os.getenv("HYBRID_ENGINE_USE_CELERY", "0") == "1"
    )


def _dispatch_or_thread(task_name: str, target: Callable[..., Any], args: tuple[Any, ...]) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs import tasks

        task = getattr(tasks, task_name, None)
        if task is None:
            if EngineConfig.from_env().require_celery:
                raise RuntimeError(f"Celery es obligatorio, pero falta la task {task_name}.")
            return _start_local_thread(target, args)
        _ensure_celery_available()
        task.delay(*args)
        return "celery"
    return _start_local_thread(target, args)


def _ensure_celery_available() -> None:
    config = EngineConfig.from_env()
    if not config.require_celery:
        return
    try:
        from hybrid_music_engine.jobs.celery_app import celery_app

        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=1)
    except Exception as exc:
        raise RuntimeError(
            "Celery/Redis es obligatorio en producción, pero no se pudo conectar al broker "
            f"{config.celery_broker_url}: {exc}"
        ) from exc


def _run_with_failure_status(target: Callable[..., Any], args: tuple[Any, ...]) -> None:
    job_id = str(args[0])
    try:
        target(*args)
    except Exception as exc:
        config = EngineConfig.from_env()
        update_job(
            config,
            job_id,
            status="failed",
            stage="Error",
            message=str(exc),
            error=str(exc),
        )
        raise


def _start_local_thread(target: Callable[..., Any], args: tuple[Any, ...]) -> str:
    thread = threading.Thread(
        target=_run_with_failure_status,
        args=(target, args),
        daemon=True,
    )
    thread.start()
    return "local-thread"


def dispatch_import_audio(job_id: str, project_id: str, source_path: str) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import import_audio_task

        _ensure_celery_available()
        import_audio_task.delay(job_id, project_id, source_path)
        return "celery"

    return _start_local_thread(run_import_audio_job, (job_id, project_id, source_path))


def dispatch_separate_stems(
    job_id: str,
    project_id: str,
    audio_path: str | None = None,
    model_name: str = "htdemucs",
    device: str = "auto",
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import separate_stems_task

        _ensure_celery_available()
        separate_stems_task.delay(job_id, project_id, audio_path, model_name, device)
        return "celery"

    return _start_local_thread(
        run_separate_stems_job,
        (job_id, project_id, audio_path, model_name, device),
    )


def dispatch_transcribe_melodic(
    job_id: str,
    project_id: str,
    stems: list[str] | None = None,
    audio_path: str | None = None,
    minimum_note_length: float | None = None,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import transcribe_melodic_task

        _ensure_celery_available()
        transcribe_melodic_task.delay(
            job_id,
            project_id,
            stems,
            audio_path,
            minimum_note_length,
            onset_threshold,
            frame_threshold,
        )
        return "celery"

    return _start_local_thread(
        run_transcribe_melodic_job,
        (
            job_id,
            project_id,
            stems,
            audio_path,
            minimum_note_length,
            onset_threshold,
            frame_threshold,
        ),
    )


def dispatch_transcribe_drums(
    job_id: str,
    project_id: str,
    audio_path: str | None = None,
    bpm: float | None = None,
    onset_delta: float = 0.07,
    onset_wait: float = 0.03,
    note_length: float = 0.08,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import transcribe_drums_task

        _ensure_celery_available()
        transcribe_drums_task.delay(
            job_id,
            project_id,
            audio_path,
            bpm,
            onset_delta,
            onset_wait,
            note_length,
        )
        return "celery"

    return _start_local_thread(
        run_transcribe_drums_job,
        (
            job_id,
            project_id,
            audio_path,
            bpm,
            onset_delta,
            onset_wait,
            note_length,
        ),
    )


def dispatch_extract_features(
    job_id: str,
    project_id: str,
    include_audio: bool = True,
    include_midis: bool = True,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import extract_features_task

        _ensure_celery_available()
        extract_features_task.delay(job_id, project_id, include_audio, include_midis)
        return "celery"

    return _start_local_thread(
        run_extract_features_job,
        (job_id, project_id, include_audio, include_midis),
    )


def dispatch_train_vae(
    job_id: str,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import train_vae_task

        _ensure_celery_available()
        train_vae_task.delay(job_id, latent_dim, hidden_dim, epochs, learning_rate, beta, seed)
        return "celery"

    return _start_local_thread(
        run_train_vae_job,
        (job_id, latent_dim, hidden_dim, epochs, learning_rate, beta, seed),
    )


def dispatch_encode_project(
    job_id: str,
    project_id: str,
    model_path: str | None = None,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import encode_project_task

        _ensure_celery_available()
        encode_project_task.delay(job_id, project_id, model_path)
        return "celery"

    return _start_local_thread(
        run_encode_project_job,
        (job_id, project_id, model_path),
    )


def dispatch_download_jamendo(
    job_id: str,
    genre_tags: dict[str, list[str]] | None = None,
    catalog_name: str = "mtg_jamendo",
    tracks_per_page: int = 200,
    max_tracks_per_genre: int | None = 500,
    download_audio: bool = True,
    client_id: str = "b6747d04",
    source: str = "mtg-cdn",
    concurrent_downloads: int = 16,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import download_jamendo_task

        _ensure_celery_available()
        download_jamendo_task.delay(
            job_id,
            genre_tags,
            catalog_name,
            tracks_per_page,
            max_tracks_per_genre,
            download_audio,
            client_id,
            source,
            concurrent_downloads,
        )
        return "celery"

    return _start_local_thread(
        run_download_jamendo_job,
        (
            job_id,
            genre_tags,
            catalog_name,
            tracks_per_page,
            max_tracks_per_genre,
            download_audio,
            client_id,
            source,
            concurrent_downloads,
        ),
    )


def dispatch_prepare_jamendo_clips(
    job_id: str,
    catalog_path: str,
    clip_duration_seconds: float = 20.0,
    hop_duration_seconds: float | None = None,
    max_clips_per_track: int | None = None,
    min_clip_seconds: float = 5.0,
    sample_rate: int | None = None,
    mono: bool = True,
) -> str:
    if _use_celery():
        from hybrid_music_engine.jobs.tasks import prepare_jamendo_clips_task

        _ensure_celery_available()
        prepare_jamendo_clips_task.delay(
            job_id,
            catalog_path,
            clip_duration_seconds,
            hop_duration_seconds,
            max_clips_per_track,
            min_clip_seconds,
            sample_rate,
            mono,
        )
        return "celery"

    return _start_local_thread(
        run_prepare_jamendo_clips_job,
        (
            job_id,
            catalog_path,
            clip_duration_seconds,
            hop_duration_seconds,
            max_clips_per_track,
            min_clip_seconds,
            sample_rate,
            mono,
        ),
    )


def dispatch_process_jamendo_clips(
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
) -> str:
    return _dispatch_or_thread(
        "process_jamendo_clips_task",
        run_process_jamendo_clips_job,
        (
            job_id,
            clips_catalog_path,
            max_clips,
            run_stems,
            run_melodic,
            run_drums,
            run_features,
            run_tokens,
            continue_on_error,
            processing_mode,
            midi_cleanup,
            quantize_grid,
            strict_demucs,
        ),
    )


def dispatch_train_token_model(
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
) -> str:
    return _dispatch_or_thread(
        "train_token_model_task",
        run_train_token_model_job,
        (
            job_id,
            token_manifest_path,
            model_name,
            order,
            model_type,
            sequence_length,
            epochs,
            batch_size,
            embedding_dim,
            num_layers,
            num_heads,
        ),
    )


def dispatch_generate_tokens(
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
) -> str:
    return _dispatch_or_thread(
        "generate_tokens_task",
        run_generate_tokens_job,
        (
            job_id,
            model_path,
            duration_seconds,
            output_name,
            seed,
            max_tokens,
            temperature,
            top_k,
            top_p,
            condition_genre,
            feature_tokens,
            embedding_path,
            token_vae_embedding_path,
            export_layers,
        ),
    )


def dispatch_generate_ranked(
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
) -> str:
    return _dispatch_or_thread(
        "generate_ranked_task",
        run_generate_ranked_job,
        (
            job_id,
            model_path,
            duration_seconds,
            output_name,
            candidates,
            seed,
            max_tokens,
            temperature,
            top_k,
            top_p,
            condition_genre,
            feature_tokens,
            embedding_path,
            token_vae_embedding_path,
            export_layers,
            render_best,
            render_engine,
            soundfont_path,
            export_mp3,
        ),
    )


def dispatch_blend_embeddings(
    job_id: str,
    embedding_a_path: str,
    embedding_b_path: str,
    alpha: float = 0.5,
    output_name: str = "latent_blend",
) -> str:
    return _dispatch_or_thread(
        "blend_embeddings_task",
        run_blend_embeddings_job,
        (job_id, embedding_a_path, embedding_b_path, alpha, output_name),
    )


def dispatch_blend_weighted_embeddings(
    job_id: str,
    embeddings: list[dict],
    output_name: str = "genre_fusion",
) -> str:
    return _dispatch_or_thread(
        "blend_weighted_embeddings_task",
        run_blend_weighted_embeddings_job,
        (job_id, embeddings, output_name),
    )


def dispatch_compare_fusions(
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
) -> str:
    return _dispatch_or_thread(
        "compare_fusions_task",
        run_compare_fusions_job,
        (
            job_id,
            model_path,
            fusion_embeddings,
            duration_seconds,
            output_name,
            candidates_per_fusion,
            seed,
            max_tokens,
            temperature,
            top_k,
            top_p,
            feature_tokens,
            export_layers,
            render_best,
            render_engine,
            soundfont_path,
            export_mp3,
        ),
    )


def dispatch_render_midi(
    job_id: str,
    midi_path: str,
    output_name: str = "preview",
    engine: str = "auto",
    soundfont_path: str | None = None,
    sample_rate: int = 44100,
    export_mp3: bool = False,
    pedalboard_preset: str = "master",
    plugin_paths: list[str] | None = None,
) -> str:
    return _dispatch_or_thread(
        "render_midi_task",
        run_render_midi_job,
        (
            job_id,
            midi_path,
            output_name,
            engine,
            soundfont_path,
            sample_rate,
            export_mp3,
            pedalboard_preset,
            plugin_paths,
        ),
    )


def dispatch_render_layers(
    job_id: str,
    generation_path: str,
    output_name: str = "layer_render",
    engine: str = "auto",
    soundfont_path: str | None = None,
    sample_rate: int = 44100,
    export_mp3: bool = False,
    pedalboard_preset: str = "master",
    plugin_paths: list[str] | None = None,
) -> str:
    return _dispatch_or_thread(
        "render_layers_task",
        run_render_layers_job,
        (
            job_id,
            generation_path,
            output_name,
            engine,
            soundfont_path,
            sample_rate,
            export_mp3,
            pedalboard_preset,
            plugin_paths,
        ),
    )


def dispatch_midi_metrics(job_id: str, midi_path: str) -> str:
    return _dispatch_or_thread("midi_metrics_task", run_midi_metrics_job, (job_id, midi_path))


def dispatch_augment_midi(
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
) -> str:
    return _dispatch_or_thread(
        "augment_midi_task",
        run_augment_midi_job,
        (
            job_id,
            catalog_path,
            source_dir,
            output_name,
            transpose_steps,
            velocity_jitter,
            timing_jitter_ticks,
            quantize_step_ticks,
            tempo_scale,
            seed,
        ),
    )


def dispatch_train_token_vae(
    job_id: str,
    token_manifest_path: str,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> str:
    return _dispatch_or_thread(
        "train_token_vae_task",
        run_train_token_vae_job,
        (job_id, token_manifest_path, latent_dim, hidden_dim, epochs, learning_rate, beta, seed),
    )


def dispatch_encode_token_vae(
    job_id: str,
    token_source_path: str,
    model_path: str | None = None,
    output_name: str = "token_embedding",
) -> str:
    return _dispatch_or_thread(
        "encode_token_vae_task",
        run_encode_token_vae_job,
        (job_id, token_source_path, model_path, output_name),
    )


def dispatch_encode_genre_embeddings(
    job_id: str,
    token_manifest_path: str,
    model_path: str | None = None,
    output_name: str = "genre_embeddings",
) -> str:
    return _dispatch_or_thread(
        "encode_genre_embeddings_task",
        run_encode_genre_embeddings_job,
        (job_id, token_manifest_path, model_path, output_name),
    )



def dispatch_evaluation_generate_batch(
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
) -> str:
    from hybrid_music_engine.jobs.workflows import run_evaluation_generate_batch_job
    return _dispatch_or_thread(
        "evaluation_generate_batch_task",
        run_evaluation_generate_batch_job,
        (
            job_id,
            model_path,
            distribution,
            real_audio_root,
            duration_seconds,
            output_name,
            seed,
            max_tokens,
            temperature,
            top_k,
            top_p,
            export_layers,
            render_audio,
            render_engine,
            export_mp3,
            target_total,
        ),
    )


def dispatch_evaluation_run(
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
) -> str:
    from hybrid_music_engine.jobs.workflows import run_evaluation_run_job
    return _dispatch_or_thread(
        "evaluation_run_task",
        run_evaluation_run_job,
        (
            job_id,
            evaluation_id,
            generated_root,
            real_root,
            prompts_path,
            classifier_path,
            train_classifier_if_missing,
            metrics,
            fad_extractor,
            clap_model,
            device,
        ),
    )


def dispatch_evaluation_from_results(
    job_id: str,
    selections: list[dict],
    genre_selections: dict[str, list[dict]] | None = None,
    target_per_genre: int = 20,
    pairing_strategy: str = "same_genre_round_robin",
    real_audio_root: str | None = None,
    output_name: str = "generated_results",
    metrics: list[str] | None = None,
) -> str:
    from hybrid_music_engine.jobs.workflows import run_evaluation_from_results_job
    return _dispatch_or_thread(
        "evaluation_from_results_task",
        run_evaluation_from_results_job,
        (
            job_id,
            selections,
            genre_selections or {},
            target_per_genre,
            pairing_strategy,
            real_audio_root,
            output_name,
            metrics,
        ),
    )


def dispatch_generate_pretrained(job_id: str, model_name: str) -> str:
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[3] / "scripts" / "generate_all.py"

    def _run() -> None:
        update_job(job_id, status="running", stage="generate-pretrained",
                   message=f"Iniciando {model_name}…", progress=0.0)
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "--models", model_name],
                capture_output=True, text=True,
            )
            if proc.returncode == 0:
                update_job(job_id, status="completed", stage="generate-pretrained",
                           message=f"{model_name} completado.", progress=1.0,
                           payload={"stdout": proc.stdout[-3000:]})
            else:
                update_job(job_id, status="failed", stage="generate-pretrained",
                           message=(proc.stderr or proc.stdout or "Error desconocido")[-1000:],
                           progress=0.0)
        except Exception as exc:
            update_job(job_id, status="failed", stage="generate-pretrained",
                       message=str(exc), progress=0.0)

    return _start_local_thread(_run, ())


def dispatch_classifier_train(
    job_id: str,
    real_audio_root: str,
    labels: list[str] | None = None,
    output_name: str = "audio_classifier",
    max_files_per_class: int | None = None,
    temperature: float = 1.0,
) -> str:
    from hybrid_music_engine.jobs.workflows import run_classifier_train_job
    return _dispatch_or_thread(
        "classifier_train_task",
        run_classifier_train_job,
        (job_id, real_audio_root, labels, output_name, max_files_per_class, temperature),
    )


def dispatch_classifier_predict(
    job_id: str,
    model_path: str | None = None,
    audio_paths: list[str] | None = None,
    audio_root: str | None = None,
) -> str:
    from hybrid_music_engine.jobs.workflows import run_classifier_predict_job
    return _dispatch_or_thread(
        "classifier_predict_task",
        run_classifier_predict_job,
        (job_id, model_path, audio_paths, audio_root),
    )
