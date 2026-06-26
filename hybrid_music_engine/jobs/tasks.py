from __future__ import annotations

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.jobs.celery_app import celery_app
from hybrid_music_engine.storage.job_store import update_job
from hybrid_music_engine.jobs.workflows import (
    run_augment_midi_job,
    run_blend_embeddings_job,
    run_blend_weighted_embeddings_job,
    run_compare_fusions_job,
    run_encode_genre_embeddings_job,
    run_encode_token_vae_job,
    run_extract_features_job,
    run_encode_project_job,
    run_download_jamendo_job,
    run_generate_ranked_job,
    run_generate_tokens_job,
    run_import_audio_job,
    run_midi_metrics_job,
    run_prepare_jamendo_clips_job,
    run_process_jamendo_clips_job,
    run_render_layers_job,
    run_render_midi_job,
    run_separate_stems_job,
    run_train_token_model_job,
    run_train_token_vae_job,
    run_train_vae_job,
    run_transcribe_drums_job,
    run_transcribe_melodic_job,
)


@celery_app.task(name="hybrid.import_audio")
def import_audio_task(job_id: str, project_id: str, source_path: str) -> dict:
    try:
        return run_import_audio_job(job_id, project_id, source_path)
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


@celery_app.task(name="hybrid.separate_stems")
def separate_stems_task(
    job_id: str,
    project_id: str,
    audio_path: str | None = None,
    model_name: str = "htdemucs",
    device: str = "auto",
) -> dict:
    try:
        return run_separate_stems_job(job_id, project_id, audio_path, model_name, device)
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


@celery_app.task(name="hybrid.transcribe_melodic")
def transcribe_melodic_task(
    job_id: str,
    project_id: str,
    stems: list[str] | None = None,
    audio_path: str | None = None,
    minimum_note_length: float | None = None,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
) -> dict:
    try:
        return run_transcribe_melodic_job(
            job_id,
            project_id,
            stems,
            audio_path,
            minimum_note_length,
            onset_threshold,
            frame_threshold,
        )
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


@celery_app.task(name="hybrid.transcribe_drums")
def transcribe_drums_task(
    job_id: str,
    project_id: str,
    audio_path: str | None = None,
    bpm: float | None = None,
    onset_delta: float = 0.07,
    onset_wait: float = 0.03,
    note_length: float = 0.08,
) -> dict:
    try:
        return run_transcribe_drums_job(
            job_id,
            project_id,
            audio_path,
            bpm,
            onset_delta,
            onset_wait,
            note_length,
        )
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


@celery_app.task(name="hybrid.extract_features")
def extract_features_task(
    job_id: str,
    project_id: str,
    include_audio: bool = True,
    include_midis: bool = True,
) -> dict:
    try:
        return run_extract_features_job(job_id, project_id, include_audio, include_midis)
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


@celery_app.task(name="hybrid.train_vae")
def train_vae_task(
    job_id: str,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> dict:
    try:
        return run_train_vae_job(
            job_id,
            latent_dim,
            hidden_dim,
            epochs,
            learning_rate,
            beta,
            seed,
        )
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


@celery_app.task(name="hybrid.encode_project")
def encode_project_task(
    job_id: str,
    project_id: str,
    model_path: str | None = None,
) -> dict:
    try:
        return run_encode_project_job(job_id, project_id, model_path)
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


@celery_app.task(name="hybrid.download_jamendo")
def download_jamendo_task(
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
    try:
        return run_download_jamendo_job(
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


@celery_app.task(name="hybrid.prepare_jamendo_clips")
def prepare_jamendo_clips_task(
    job_id: str,
    catalog_path: str,
    clip_duration_seconds: float = 20.0,
    hop_duration_seconds: float | None = None,
    max_clips_per_track: int | None = None,
    min_clip_seconds: float = 5.0,
    sample_rate: int | None = None,
    mono: bool = True,
) -> dict:
    try:
        return run_prepare_jamendo_clips_job(
            job_id,
            catalog_path,
            clip_duration_seconds,
            hop_duration_seconds,
            max_clips_per_track,
            min_clip_seconds,
            sample_rate,
            mono,
        )
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


@celery_app.task(name="hybrid.process_jamendo_clips")
def process_jamendo_clips_task(
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
    return _run_task(
        run_process_jamendo_clips_job,
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
    )


@celery_app.task(name="hybrid.train_token_model")
def train_token_model_task(
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
    return _run_task(
        run_train_token_model_job,
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
    )


@celery_app.task(name="hybrid.generate_tokens")
def generate_tokens_task(
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
    return _run_task(
        run_generate_tokens_job,
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
    )


@celery_app.task(name="hybrid.generate_ranked")
def generate_ranked_task(
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
    return _run_task(
        run_generate_ranked_job,
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
    )


@celery_app.task(name="hybrid.render_midi")
def render_midi_task(
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
    return _run_task(
        run_render_midi_job,
        job_id,
        midi_path,
        output_name,
        engine,
        soundfont_path,
        sample_rate,
        export_mp3,
        pedalboard_preset,
        plugin_paths,
    )


@celery_app.task(name="hybrid.render_layers")
def render_layers_task(
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
    return _run_task(
        run_render_layers_job,
        job_id,
        generation_path,
        output_name,
        engine,
        soundfont_path,
        sample_rate,
        export_mp3,
        pedalboard_preset,
        plugin_paths,
    )


@celery_app.task(name="hybrid.midi_metrics")
def midi_metrics_task(job_id: str, midi_path: str) -> dict:
    return _run_task(run_midi_metrics_job, job_id, midi_path)


@celery_app.task(name="hybrid.blend_embeddings")
def blend_embeddings_task(
    job_id: str,
    embedding_a_path: str,
    embedding_b_path: str,
    alpha: float = 0.5,
    output_name: str = "latent_blend",
) -> dict:
    return _run_task(
        run_blend_embeddings_job,
        job_id,
        embedding_a_path,
        embedding_b_path,
        alpha,
        output_name,
    )


@celery_app.task(name="hybrid.blend_weighted_embeddings")
def blend_weighted_embeddings_task(
    job_id: str,
    embeddings: list[dict],
    output_name: str = "genre_fusion",
) -> dict:
    return _run_task(
        run_blend_weighted_embeddings_job,
        job_id,
        embeddings,
        output_name,
    )


@celery_app.task(name="hybrid.compare_fusions")
def compare_fusions_task(
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
    return _run_task(
        run_compare_fusions_job,
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
    )


@celery_app.task(name="hybrid.augment_midi")
def augment_midi_task(
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
    return _run_task(
        run_augment_midi_job,
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
    )


@celery_app.task(name="hybrid.train_token_vae")
def train_token_vae_task(
    job_id: str,
    token_manifest_path: str,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> dict:
    return _run_task(
        run_train_token_vae_job,
        job_id,
        token_manifest_path,
        latent_dim,
        hidden_dim,
        epochs,
        learning_rate,
        beta,
        seed,
    )


@celery_app.task(name="hybrid.encode_token_vae")
def encode_token_vae_task(
    job_id: str,
    token_source_path: str,
    model_path: str | None = None,
    output_name: str = "token_embedding",
) -> dict:
    return _run_task(
        run_encode_token_vae_job,
        job_id,
        token_source_path,
        model_path,
        output_name,
    )


@celery_app.task(name="hybrid.encode_genre_embeddings")
def encode_genre_embeddings_task(
    job_id: str,
    token_manifest_path: str,
    model_path: str | None = None,
    output_name: str = "genre_embeddings",
) -> dict:
    return _run_task(
        run_encode_genre_embeddings_job,
        job_id,
        token_manifest_path,
        model_path,
        output_name,
    )


def _run_task(target, *args):
    try:
        return target(*args)
    except Exception as exc:
        config = EngineConfig.from_env()
        update_job(
            config,
            str(args[0]),
            status="failed",
            stage="Error",
            message=str(exc),
            error=str(exc),
        )
        raise



@celery_app.task(name="hybrid.evaluation_generate_batch")
def evaluation_generate_batch_task(
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
    from hybrid_music_engine.jobs.workflows import run_evaluation_generate_batch_job
    return _run_task(
        run_evaluation_generate_batch_job,
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
    )


@celery_app.task(name="hybrid.evaluation_run")
def evaluation_run_task(
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
    from hybrid_music_engine.jobs.workflows import run_evaluation_run_job
    return _run_task(
        run_evaluation_run_job,
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
    )


@celery_app.task(name="hybrid.evaluation_from_results")
def evaluation_from_results_task(
    job_id: str,
    selections: list[dict],
    genre_selections: dict[str, list[dict]] | None = None,
    target_per_genre: int = 20,
    pairing_strategy: str = "same_genre_round_robin",
    real_audio_root: str | None = None,
    output_name: str = "generated_results",
    metrics: list[str] | None = None,
) -> dict:
    from hybrid_music_engine.jobs.workflows import run_evaluation_from_results_job
    return _run_task(
        run_evaluation_from_results_job,
        job_id,
        selections,
        genre_selections or {},
        target_per_genre,
        pairing_strategy,
        real_audio_root,
        output_name,
        metrics,
    )


@celery_app.task(name="hybrid.classifier_train")
def classifier_train_task(
    job_id: str,
    real_audio_root: str,
    labels: list[str] | None = None,
    output_name: str = "audio_classifier",
    max_files_per_class: int | None = None,
    temperature: float = 1.0,
) -> dict:
    from hybrid_music_engine.jobs.workflows import run_classifier_train_job
    return _run_task(
        run_classifier_train_job,
        job_id,
        real_audio_root,
        labels,
        output_name,
        max_files_per_class,
        temperature,
    )


@celery_app.task(name="hybrid.classifier_predict")
def classifier_predict_task(
    job_id: str,
    model_path: str | None = None,
    audio_paths: list[str] | None = None,
    audio_root: str | None = None,
) -> dict:
    from hybrid_music_engine.jobs.workflows import run_classifier_predict_job
    return _run_task(run_classifier_predict_job, job_id, model_path, audio_paths, audio_root)
