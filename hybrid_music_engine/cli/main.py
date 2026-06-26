from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.presets import presets_payload
from hybrid_music_engine.datasets.cleaning import clean_midi_dataset
from hybrid_music_engine.datasets.genre_catalog import build_genre_catalog
from hybrid_music_engine.jobs.dispatcher import (
    dispatch_augment_midi,
    dispatch_blend_embeddings,
    dispatch_blend_weighted_embeddings,
    dispatch_compare_fusions,
    dispatch_encode_project,
    dispatch_encode_genre_embeddings,
    dispatch_encode_token_vae,
    dispatch_download_jamendo,
    dispatch_extract_features,
    dispatch_generate_tokens,
    dispatch_generate_ranked,
    dispatch_import_audio,
    dispatch_prepare_jamendo_clips,
    dispatch_process_jamendo_clips,
    dispatch_render_layers,
    dispatch_render_midi,
    dispatch_separate_stems,
    dispatch_train_token_model,
    dispatch_train_token_vae,
    dispatch_train_vae,
    dispatch_transcribe_drums,
    dispatch_transcribe_melodic,
)
from hybrid_music_engine.storage.job_store import create_job, list_jobs, load_job
from hybrid_music_engine.storage.catalog import (
    list_blends,
    list_fusion_comparisons,
    list_generations,
    list_renders,
    list_rankings,
    list_token_manifests,
    list_token_models,
    list_token_vae_assets,
)
from hybrid_music_engine.storage.manifest import create_project, list_projects, load_manifest
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
from hybrid_music_engine.tokens.midi_tokenizer import (
    create_generation_plan,
    export_token_manifest_to_zip,
    export_output_tokens_to_zip,
    tokenize_catalog_to_zip,
)


app = typer.Typer(help="Motor de transformación musical híbrida.")
config = EngineConfig.from_env()


@app.command("create-project")
def create_project_command(name: Annotated[str, typer.Option("--name")] = "demo") -> None:
    manifest = create_project(config, name)
    typer.echo(asdict(manifest))


@app.command("projects")
def projects_command() -> None:
    typer.echo({"projects": list_projects(config)})


@app.command("project")
def project_command(project_id: str) -> None:
    typer.echo(asdict(load_manifest(config, project_id)))


@app.command("import-audio")
def import_audio_command(
    project_id: Annotated[str, typer.Option("--project-id")],
    source_path: Annotated[Path, typer.Option("--source")],
) -> None:
    job = create_job(config, kind="import-audio", project_id=project_id)
    mode = dispatch_import_audio(job.job_id, project_id, str(source_path))
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("separate-stems")
def separate_stems_command(
    project_id: Annotated[str, typer.Option("--project-id")],
    audio_path: Annotated[Path | None, typer.Option("--audio")] = None,
    model_name: Annotated[str, typer.Option("--model")] = "htdemucs",
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    job = create_job(config, kind="separate-stems", project_id=project_id)
    mode = dispatch_separate_stems(
        job.job_id,
        project_id,
        str(audio_path) if audio_path else None,
        model_name,
        device,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("transcribe-melodic")
def transcribe_melodic_command(
    project_id: Annotated[str, typer.Option("--project-id")],
    stems: Annotated[str, typer.Option("--stems")] = "bass,vocals,other",
    audio_path: Annotated[Path | None, typer.Option("--audio")] = None,
    minimum_note_length: Annotated[float | None, typer.Option("--minimum-note-length")] = None,
    onset_threshold: Annotated[float | None, typer.Option("--onset-threshold")] = None,
    frame_threshold: Annotated[float | None, typer.Option("--frame-threshold")] = None,
) -> None:
    selected_stems = [item.strip() for item in stems.split(",") if item.strip()]
    job = create_job(config, kind="transcribe-melodic", project_id=project_id)
    mode = dispatch_transcribe_melodic(
        job.job_id,
        project_id,
        selected_stems,
        str(audio_path) if audio_path else None,
        minimum_note_length,
        onset_threshold,
        frame_threshold,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("transcribe-drums")
def transcribe_drums_command(
    project_id: Annotated[str, typer.Option("--project-id")],
    audio_path: Annotated[Path | None, typer.Option("--audio")] = None,
    bpm: Annotated[float | None, typer.Option("--bpm")] = None,
    onset_delta: Annotated[float, typer.Option("--onset-delta")] = 0.07,
    onset_wait: Annotated[float, typer.Option("--onset-wait")] = 0.03,
    note_length: Annotated[float, typer.Option("--note-length")] = 0.08,
) -> None:
    job = create_job(config, kind="transcribe-drums", project_id=project_id)
    mode = dispatch_transcribe_drums(
        job.job_id,
        project_id,
        str(audio_path) if audio_path else None,
        bpm,
        onset_delta,
        onset_wait,
        note_length,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("extract-features")
def extract_features_command(
    project_id: Annotated[str, typer.Option("--project-id")],
    include_audio: Annotated[bool, typer.Option("--audio/--no-audio")] = True,
    include_midis: Annotated[bool, typer.Option("--midis/--no-midis")] = True,
) -> None:
    job = create_job(config, kind="extract-features", project_id=project_id)
    mode = dispatch_extract_features(
        job.job_id,
        project_id,
        include_audio,
        include_midis,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("train-vae")
def train_vae_command(
    latent_dim: Annotated[int, typer.Option("--latent-dim")] = 32,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim")] = 128,
    epochs: Annotated[int, typer.Option("--epochs")] = 200,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
    beta: Annotated[float, typer.Option("--beta")] = 0.001,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    job = create_job(config, kind="train-vae")
    mode = dispatch_train_vae(
        job.job_id,
        latent_dim,
        hidden_dim,
        epochs,
        learning_rate,
        beta,
        seed,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("encode-project")
def encode_project_command(
    project_id: Annotated[str, typer.Option("--project-id")],
    model_path: Annotated[Path | None, typer.Option("--model")] = None,
) -> None:
    job = create_job(config, kind="encode-project", project_id=project_id)
    mode = dispatch_encode_project(
        job.job_id,
        project_id,
        str(model_path) if model_path else None,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("build-genre-catalog")
def build_genre_catalog_command(
    source_dir: Annotated[Path, typer.Option("--source-dir")],
    genres: Annotated[str, typer.Option("--genres")],
    clips_per_genre: Annotated[int, typer.Option("--clips-per-genre")] = 200,
    max_duration_seconds: Annotated[float, typer.Option("--max-duration")] = 10.0,
    catalog_name: Annotated[str, typer.Option("--name")] = "genre_catalog",
    source_label: Annotated[str, typer.Option("--source-label")] = "source_3",
) -> None:
    payload = build_genre_catalog(
        config,
        source_dir=source_dir,
        genres=[item.strip() for item in genres.split(",") if item.strip()],
        clips_per_genre=clips_per_genre,
        max_duration_seconds=max_duration_seconds,
        catalog_name=catalog_name,
        source_label=source_label,
    )
    typer.echo(payload)


@app.command("clean-midi-dataset")
def clean_midi_dataset_command(
    source_dir: Annotated[Path, typer.Option("--source-dir")],
    output_name: Annotated[str, typer.Option("--name")] = "clean_midis",
    min_duration_seconds: Annotated[float, typer.Option("--min-duration")] = 1.0,
    max_duration_seconds: Annotated[float, typer.Option("--max-duration")] = 240.0,
    min_notes: Annotated[int, typer.Option("--min-notes")] = 4,
    min_quality_score: Annotated[float, typer.Option("--min-quality")] = 0.05,
    deduplicate: Annotated[bool, typer.Option("--deduplicate/--no-deduplicate")] = True,
) -> None:
    typer.echo(
        clean_midi_dataset(
            config,
            source_dir=source_dir,
            output_name=output_name,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            min_notes=min_notes,
            min_quality_score=min_quality_score,
            deduplicate=deduplicate,
        )
    )


@app.command("augment-midi")
def augment_midi_command(
    catalog_path: Annotated[Path | None, typer.Option("--catalog")] = None,
    source_dir: Annotated[Path | None, typer.Option("--source-dir")] = None,
    output_name: Annotated[str, typer.Option("--name")] = "augmented_midis",
    transpose_steps: Annotated[str, typer.Option("--transpose")] = "-2,0,2",
    velocity_jitter: Annotated[int, typer.Option("--velocity-jitter")] = 8,
    timing_jitter_ticks: Annotated[int, typer.Option("--timing-jitter")] = 12,
    quantize_step_ticks: Annotated[int | None, typer.Option("--quantize")] = None,
    tempo_scale: Annotated[float, typer.Option("--tempo-scale")] = 1.0,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    job = create_job(config, kind="augment-midi")
    mode = dispatch_augment_midi(
        job.job_id,
        str(catalog_path) if catalog_path else None,
        str(source_dir) if source_dir else None,
        output_name,
        [int(item.strip()) for item in transpose_steps.split(",") if item.strip()],
        velocity_jitter,
        timing_jitter_ticks,
        quantize_step_ticks,
        tempo_scale,
        seed,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("tokenize-catalog")
def tokenize_catalog_command(
    catalog_path: Annotated[Path, typer.Option("--catalog")],
    token_set_name: Annotated[str, typer.Option("--name")] = "input_tokens",
) -> None:
    typer.echo(
        tokenize_catalog_to_zip(
            config,
            catalog_path=catalog_path,
            token_set_name=token_set_name,
        )
    )


@app.command("export-input-tokens")
def export_input_tokens_command(
    token_manifest_path: Annotated[Path, typer.Option("--tokens-manifest")],
    export_name: Annotated[str, typer.Option("--name")] = "input_tokens",
) -> None:
    typer.echo(
        export_token_manifest_to_zip(
            config,
            token_manifest_path=token_manifest_path,
            export_name=export_name,
        )
    )


@app.command("export-output-tokens")
def export_output_tokens_command(
    source_dir: Annotated[Path, typer.Option("--source-dir")],
    export_name: Annotated[str, typer.Option("--name")] = "mixed_output_tokens",
    duration_seconds: Annotated[float | None, typer.Option("--duration")] = None,
) -> None:
    typer.echo(
        export_output_tokens_to_zip(
            config,
            source_dir=source_dir,
            export_name=export_name,
            duration_seconds=duration_seconds,
        )
    )


@app.command("generation-plan")
def generation_plan_command(
    duration_seconds: Annotated[float, typer.Option("--duration")],
    project_id: Annotated[str | None, typer.Option("--project-id")] = None,
    output_name: Annotated[str, typer.Option("--name")] = "generated_track",
) -> None:
    typer.echo(
        create_generation_plan(
            config,
            project_id=project_id,
            duration_seconds=duration_seconds,
            output_name=output_name,
        )
    )


@app.command("download-jamendo")
def download_jamendo_command(
    genre_tags_json: Annotated[
        Path | None,
        typer.Option("--genre-tags-json", help="JSON con {'genero': ['tag1', 'tag2']}"),
    ] = None,
    catalog_name: Annotated[str, typer.Option("--name")] = "mtg_jamendo",
    tracks_per_page: Annotated[int, typer.Option("--tracks-per-page")] = 200,
    max_tracks_per_genre: Annotated[int | None, typer.Option("--max-tracks-per-genre")] = 500,
    download_audio: Annotated[bool, typer.Option("--download-audio/--metadata-only")] = True,
    client_id: Annotated[str, typer.Option("--client-id")] = "b6747d04",
    source: Annotated[str, typer.Option("--source")] = "mtg-cdn",
    concurrent_downloads: Annotated[int, typer.Option("--concurrent-downloads")] = 16,
) -> None:
    genre_tags = None
    if genre_tags_json:
        genre_tags = json.loads(genre_tags_json.read_text(encoding="utf-8"))
    job = create_job(config, kind="download-jamendo")
    mode = dispatch_download_jamendo(
        job.job_id,
        genre_tags,
        catalog_name,
        tracks_per_page,
        max_tracks_per_genre,
        download_audio,
        client_id,
        source,
        concurrent_downloads,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("prepare-jamendo-clips")
def prepare_jamendo_clips_command(
    catalog_path: Annotated[Path, typer.Option("--catalog")],
    clip_duration_seconds: Annotated[float, typer.Option("--clip-duration")] = 20.0,
    hop_duration_seconds: Annotated[float | None, typer.Option("--hop-duration")] = None,
    max_clips_per_track: Annotated[int | None, typer.Option("--max-clips-per-track")] = None,
    min_clip_seconds: Annotated[float, typer.Option("--min-clip")] = 5.0,
    sample_rate: Annotated[int | None, typer.Option("--sample-rate")] = None,
    mono: Annotated[bool, typer.Option("--mono/--stereo")] = True,
) -> None:
    job = create_job(config, kind="prepare-jamendo-clips")
    mode = dispatch_prepare_jamendo_clips(
        job.job_id,
        str(catalog_path),
        clip_duration_seconds,
        hop_duration_seconds,
        max_clips_per_track,
        min_clip_seconds,
        sample_rate,
        mono,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("process-jamendo-clips")
def process_jamendo_clips_command(
    clips_catalog_path: Annotated[Path, typer.Option("--clips-catalog")],
    max_clips: Annotated[int | None, typer.Option("--max-clips")] = None,
    mode: Annotated[str, typer.Option("--mode")] = "quick",
    run_stems: Annotated[bool, typer.Option("--stems/--no-stems")] = False,
    run_melodic: Annotated[bool, typer.Option("--melodic/--no-melodic")] = True,
    run_drums: Annotated[bool, typer.Option("--drums/--no-drums")] = True,
    run_features: Annotated[bool, typer.Option("--features/--no-features")] = True,
    run_tokens: Annotated[bool, typer.Option("--tokens/--no-tokens")] = True,
    midi_cleanup: Annotated[bool, typer.Option("--midi-cleanup/--no-midi-cleanup")] = False,
    quantize_grid: Annotated[str, typer.Option("--quantize-grid")] = "1/16",
) -> None:
    job = create_job(config, kind="process-jamendo-clips")
    mode = dispatch_process_jamendo_clips(
        job.job_id,
        str(clips_catalog_path),
        max_clips,
        run_stems,
        run_melodic,
        run_drums,
        run_features,
        run_tokens,
        True,
        "token_vae_demucs" if mode == "token-vae-demucs" else mode,
        midi_cleanup,
        quantize_grid,
        mode == "token-vae-demucs",
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("train-token-model")
def train_token_model_command(
    token_manifest_path: Annotated[Path, typer.Option("--tokens-manifest")],
    model_name: Annotated[str, typer.Option("--name")] = "token_markov",
    order: Annotated[int, typer.Option("--order")] = 2,
    model_type: Annotated[str, typer.Option("--type")] = "markov",
    sequence_length: Annotated[int, typer.Option("--sequence-length")] = 128,
    epochs: Annotated[int, typer.Option("--epochs")] = 8,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    embedding_dim: Annotated[int, typer.Option("--embedding-dim")] = 128,
    num_layers: Annotated[int, typer.Option("--layers")] = 3,
    num_heads: Annotated[int, typer.Option("--heads")] = 4,
) -> None:
    job = create_job(config, kind="train-token-model")
    mode = dispatch_train_token_model(
        job.job_id,
        str(token_manifest_path),
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
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("train-token-vae")
def train_token_vae_command(
    token_manifest_path: Annotated[Path, typer.Option("--tokens-manifest")],
    latent_dim: Annotated[int, typer.Option("--latent-dim")] = 32,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim")] = 128,
    epochs: Annotated[int, typer.Option("--epochs")] = 80,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
    beta: Annotated[float, typer.Option("--beta")] = 0.001,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    job = create_job(config, kind="train-token-vae")
    mode = dispatch_train_token_vae(
        job.job_id,
        str(token_manifest_path),
        latent_dim,
        hidden_dim,
        epochs,
        learning_rate,
        beta,
        seed,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("encode-token-vae")
def encode_token_vae_command(
    token_source_path: Annotated[Path, typer.Option("--source")],
    model_path: Annotated[Path | None, typer.Option("--model")] = None,
    output_name: Annotated[str, typer.Option("--name")] = "token_embedding",
) -> None:
    job = create_job(config, kind="encode-token-vae")
    mode = dispatch_encode_token_vae(
        job.job_id,
        str(token_source_path),
        str(model_path) if model_path else None,
        output_name,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("encode-genre-embeddings")
def encode_genre_embeddings_command(
    token_manifest_path: Annotated[Path, typer.Option("--tokens-manifest")],
    model_path: Annotated[Path | None, typer.Option("--model")] = None,
    output_name: Annotated[str, typer.Option("--name")] = "genre_embeddings",
) -> None:
    job = create_job(config, kind="encode-genre-embeddings")
    mode = dispatch_encode_genre_embeddings(
        job.job_id,
        str(token_manifest_path),
        str(model_path) if model_path else None,
        output_name,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("generate-tokens")
def generate_tokens_command(
    model_path: Annotated[Path, typer.Option("--model")],
    duration_seconds: Annotated[float, typer.Option("--duration")],
    output_name: Annotated[str, typer.Option("--name")] = "generated",
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.9,
    top_k: Annotated[int | None, typer.Option("--top-k")] = 50,
    top_p: Annotated[float | None, typer.Option("--top-p")] = 0.95,
    condition_genre: Annotated[str | None, typer.Option("--genre")] = None,
    feature_tokens: Annotated[str, typer.Option("--features")] = "",
    embedding_path: Annotated[Path | None, typer.Option("--embedding")] = None,
    token_vae_embedding_path: Annotated[Path | None, typer.Option("--token-vae-embedding")] = None,
    export_layers: Annotated[bool, typer.Option("--layers/--no-layers")] = True,
) -> None:
    job = create_job(config, kind="generate-tokens")
    mode = dispatch_generate_tokens(
        job.job_id,
        str(model_path),
        duration_seconds,
        output_name,
        seed,
        max_tokens,
        temperature,
        top_k,
        top_p,
        condition_genre,
        [item.strip() for item in feature_tokens.split(",") if item.strip()],
        str(embedding_path) if embedding_path else None,
        str(token_vae_embedding_path) if token_vae_embedding_path else None,
        export_layers,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("generate-ranked")
def generate_ranked_command(
    model_path: Annotated[Path, typer.Option("--model")],
    duration_seconds: Annotated[float, typer.Option("--duration")],
    output_name: Annotated[str, typer.Option("--name")] = "ranked_generation",
    candidates: Annotated[int, typer.Option("--candidates")] = 6,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.9,
    top_k: Annotated[int | None, typer.Option("--top-k")] = 50,
    top_p: Annotated[float | None, typer.Option("--top-p")] = 0.95,
    condition_genre: Annotated[str | None, typer.Option("--genre")] = None,
    feature_tokens: Annotated[str, typer.Option("--features")] = "",
    embedding_path: Annotated[Path | None, typer.Option("--embedding")] = None,
    token_vae_embedding_path: Annotated[Path | None, typer.Option("--token-vae-embedding")] = None,
    export_layers: Annotated[bool, typer.Option("--layers/--no-layers")] = True,
    render_best: Annotated[bool, typer.Option("--render-best/--no-render-best")] = False,
    render_engine: Annotated[str, typer.Option("--render-engine")] = "auto",
    soundfont_path: Annotated[Path | None, typer.Option("--soundfont")] = None,
    export_mp3: Annotated[bool, typer.Option("--mp3/--no-mp3")] = False,
) -> None:
    job = create_job(config, kind="generate-ranked")
    mode = dispatch_generate_ranked(
        job.job_id,
        str(model_path),
        duration_seconds,
        output_name,
        candidates,
        seed,
        max_tokens,
        temperature,
        top_k,
        top_p,
        condition_genre,
        [item.strip() for item in feature_tokens.split(",") if item.strip()],
        str(embedding_path) if embedding_path else None,
        str(token_vae_embedding_path) if token_vae_embedding_path else None,
        export_layers,
        render_best,
        render_engine,
        str(soundfont_path) if soundfont_path else None,
        export_mp3,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("blend-embeddings")
def blend_embeddings_command(
    embedding_a: Annotated[Path, typer.Option("--a")],
    embedding_b: Annotated[Path, typer.Option("--b")],
    alpha: Annotated[float, typer.Option("--alpha")] = 0.5,
    output_name: Annotated[str, typer.Option("--name")] = "latent_blend",
) -> None:
    job = create_job(config, kind="blend-embeddings")
    mode = dispatch_blend_embeddings(
        job.job_id,
        str(embedding_a),
        str(embedding_b),
        alpha,
        output_name,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("blend-weighted-embeddings")
def blend_weighted_embeddings_command(
    embeddings: Annotated[str, typer.Option("--embeddings")],
    output_name: Annotated[str, typer.Option("--name")] = "genre_fusion",
) -> None:
    parsed = _parse_weighted_embeddings(embeddings)
    job = create_job(config, kind="blend-weighted-embeddings")
    mode = dispatch_blend_weighted_embeddings(job.job_id, parsed, output_name)
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("compare-fusions")
def compare_fusions_command(
    model_path: Annotated[Path, typer.Option("--model")],
    embeddings: Annotated[str, typer.Option("--embeddings")],
    duration_seconds: Annotated[float, typer.Option("--duration")] = 30,
    output_name: Annotated[str, typer.Option("--name")] = "fusion_comparison",
    candidates_per_fusion: Annotated[int, typer.Option("--candidates-per-fusion")] = 3,
    seed: Annotated[int | None, typer.Option("--seed")] = 42,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = 1200,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.84,
    top_k: Annotated[int | None, typer.Option("--top-k")] = 56,
    top_p: Annotated[float | None, typer.Option("--top-p")] = 0.92,
    feature_tokens: Annotated[str, typer.Option("--features")] = "",
    export_layers: Annotated[bool, typer.Option("--layers/--no-layers")] = True,
    render_best: Annotated[bool, typer.Option("--render-best/--no-render-best")] = False,
    render_engine: Annotated[str, typer.Option("--render-engine")] = "auto",
    soundfont_path: Annotated[Path | None, typer.Option("--soundfont")] = None,
    export_mp3: Annotated[bool, typer.Option("--mp3/--no-mp3")] = False,
) -> None:
    parsed = [
        {"embedding_path": item["path"], "label": item.get("label")}
        for item in _parse_labeled_embeddings(embeddings)
    ]
    job = create_job(config, kind="compare-fusions")
    mode = dispatch_compare_fusions(
        job.job_id,
        str(model_path),
        parsed,
        duration_seconds,
        output_name,
        candidates_per_fusion,
        seed,
        max_tokens,
        temperature,
        top_k,
        top_p,
        [item.strip() for item in feature_tokens.split(",") if item.strip()],
        export_layers,
        render_best,
        render_engine,
        str(soundfont_path) if soundfont_path else None,
        export_mp3,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("render-midi")
def render_midi_command(
    midi_path: Annotated[Path, typer.Option("--midi")],
    output_name: Annotated[str, typer.Option("--name")] = "preview",
    engine: Annotated[str, typer.Option("--engine")] = "auto",
    soundfont_path: Annotated[Path | None, typer.Option("--soundfont")] = None,
    sample_rate: Annotated[int, typer.Option("--sample-rate")] = 44100,
    export_mp3: Annotated[bool, typer.Option("--mp3/--no-mp3")] = False,
    pedalboard_preset: Annotated[str, typer.Option("--pedalboard-preset")] = "master",
    plugin_paths: Annotated[str, typer.Option("--plugins")] = "",
) -> None:
    job = create_job(config, kind="render-midi")
    mode = dispatch_render_midi(
        job.job_id,
        str(midi_path),
        output_name,
        engine,
        str(soundfont_path) if soundfont_path else None,
        sample_rate,
        export_mp3,
        pedalboard_preset,
        [item.strip() for item in plugin_paths.split(",") if item.strip()],
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("render-layers")
def render_layers_command(
    generation_path: Annotated[Path, typer.Option("--generation")],
    output_name: Annotated[str, typer.Option("--name")] = "layer_render",
    engine: Annotated[str, typer.Option("--engine")] = "auto",
    soundfont_path: Annotated[Path | None, typer.Option("--soundfont")] = None,
    sample_rate: Annotated[int, typer.Option("--sample-rate")] = 44100,
    export_mp3: Annotated[bool, typer.Option("--mp3/--no-mp3")] = False,
    pedalboard_preset: Annotated[str, typer.Option("--pedalboard-preset")] = "master",
    plugin_paths: Annotated[str, typer.Option("--plugins")] = "",
) -> None:
    job = create_job(config, kind="render-layers")
    mode = dispatch_render_layers(
        job.job_id,
        str(generation_path),
        output_name,
        engine,
        str(soundfont_path) if soundfont_path else None,
        sample_rate,
        export_mp3,
        pedalboard_preset,
        [item.strip() for item in plugin_paths.split(",") if item.strip()],
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("midi-metrics")
def midi_metrics_command(midi_path: Annotated[Path, typer.Option("--midi")]) -> None:
    typer.echo({"metrics": analyze_midi_quality(midi_path)})


@app.command("jobs")
def jobs_command() -> None:
    typer.echo({"jobs": list_jobs(config)})


@app.command("job")
def job_command(job_id: str) -> None:
    typer.echo(asdict(load_job(config, job_id)))


@app.command("resources")
def resources_command() -> None:
    typer.echo(
        {
            "token_manifests": list_token_manifests(config),
            "token_models": list_token_models(config),
            "generations": list_generations(config),
            "rankings": list_rankings(config),
            "renders": list_renders(config),
            "blends": list_blends(config),
            "fusion_comparisons": list_fusion_comparisons(config),
            "token_vae": list_token_vae_assets(config),
        }
    )


@app.command("celery-check")
def celery_check_command() -> None:
    try:
        from hybrid_music_engine.jobs.dispatcher import _ensure_celery_available

        _ensure_celery_available()
        typer.echo({"ok": True, "broker": config.celery_broker_url})
    except RuntimeError as exc:
        typer.echo({"ok": False, "broker": config.celery_broker_url, "error": str(exc)})
        raise typer.Exit(code=1) from exc


@app.command("presets")
def presets_command() -> None:
    typer.echo(presets_payload())


@app.command("token-models")
def token_models_command() -> None:
    typer.echo({"token_models": list_token_models(config)})


@app.command("generations")
def generations_command() -> None:
    typer.echo({"generations": list_generations(config)})


@app.command("renders")
def renders_command() -> None:
    typer.echo({"renders": list_renders(config)})


def _parse_weighted_embeddings(value: str) -> list[dict]:
    parsed: list[dict] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            path_value, weight_value = item.rsplit("=", 1)
        elif ":" in item:
            path_value, weight_value = item.rsplit(":", 1)
        else:
            raise typer.BadParameter("Usa path=weight o path:weight, separado por comas.")
        try:
            weight = float(weight_value)
        except ValueError as exc:
            raise typer.BadParameter(f"Peso inválido para {path_value}: {weight_value}") from exc
        parsed.append({"path": path_value.strip(), "weight": weight})
    if len(parsed) < 2:
        raise typer.BadParameter("La fusión ponderada requiere al menos dos embeddings.")
    return parsed


def _parse_labeled_embeddings(value: str) -> list[dict]:
    parsed: list[dict] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            label, path_value = item.split("=", 1)
            parsed.append({"label": label.strip(), "path": path_value.strip()})
        else:
            parsed.append({"label": None, "path": item})
    if not parsed:
        raise typer.BadParameter("Indica al menos un embedding de fusión.")
    return parsed


@app.command("reward-model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def reward_model_command(ctx: typer.Context) -> None:
    """Ejecuta comandos del reward model (bootstrap, train, score, rerank, etc.)"""
    from hybrid_music_engine.reward_model.cli import main as reward_cli_main
    import sys
    sys.exit(reward_cli_main(ctx.args))



@app.command("evaluation-availability")
def evaluation_availability_command(
    real_audio_root: Annotated[Path | None, typer.Option("--real-root")] = None,
    target_total: Annotated[int, typer.Option("--target-total")] = 100,
) -> None:
    from hybrid_music_engine.evaluation.pipeline import evaluation_availability
    typer.echo(evaluation_availability(config, real_audio_root=real_audio_root, target_total=target_total))


@app.command("evaluations")
def evaluations_command() -> None:
    from hybrid_music_engine.evaluation.pipeline import list_evaluations
    typer.echo({"evaluations": list_evaluations(config)})


@app.command("evaluation-generate-batch")
def evaluation_generate_batch_command(
    model_path: Annotated[Path, typer.Option("--model")],
    distribution: Annotated[str, typer.Option("--distribution")] = "auto",
    real_audio_root: Annotated[Path | None, typer.Option("--real-root")] = None,
    duration_seconds: Annotated[float, typer.Option("--duration")] = 30.0,
    output_name: Annotated[str, typer.Option("--name")] = "evaluation_batch",
    seed: Annotated[int | None, typer.Option("--seed")] = 42,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.9,
    top_k: Annotated[int | None, typer.Option("--top-k")] = 50,
    top_p: Annotated[float | None, typer.Option("--top-p")] = 0.95,
    export_layers: Annotated[bool, typer.Option("--layers/--no-layers")] = True,
    render_audio: Annotated[bool, typer.Option("--render-audio/--no-render-audio")] = True,
    render_engine: Annotated[str, typer.Option("--render-engine")] = "auto",
    export_mp3: Annotated[bool, typer.Option("--mp3/--no-mp3")] = True,
    target_total: Annotated[int, typer.Option("--target-total")] = 100,
) -> None:
    from hybrid_music_engine.jobs.dispatcher import dispatch_evaluation_generate_batch
    parsed_distribution = _parse_evaluation_distribution(distribution, real_audio_root, target_total)
    job = create_job(config, kind="evaluation-generate-batch")
    mode = dispatch_evaluation_generate_batch(
        job.job_id,
        str(model_path),
        parsed_distribution,
        str(real_audio_root) if real_audio_root else None,
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
    typer.echo({"job_id": job.job_id, "mode": mode, "distribution": parsed_distribution or "auto"})


@app.command("evaluation-run")
def evaluation_run_command(
    evaluation_id: Annotated[str | None, typer.Option("--evaluation-id")] = None,
    generated_root: Annotated[Path | None, typer.Option("--generated-root")] = None,
    real_root: Annotated[Path | None, typer.Option("--real-root")] = None,
    classifier_path: Annotated[Path | None, typer.Option("--classifier")] = None,
    metrics: Annotated[str, typer.Option("--metrics")] = "fad,kld,tempo,midi",
    clap: Annotated[bool, typer.Option("--clap/--no-clap")] = False,
    fad_extractor: Annotated[str, typer.Option("--fad-extractor")] = "mel",
    clap_model: Annotated[str, typer.Option("--clap-model")] = "clap",
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    from hybrid_music_engine.jobs.dispatcher import dispatch_evaluation_run
    metric_list = [item.strip() for item in metrics.split(",") if item.strip()]
    if clap and "clap" not in metric_list:
        metric_list.append("clap")
    job = create_job(config, kind="evaluation-run")
    mode = dispatch_evaluation_run(
        job.job_id,
        evaluation_id,
        str(generated_root) if generated_root else None,
        str(real_root) if real_root else None,
        None,
        str(classifier_path) if classifier_path else None,
        True,
        metric_list,
        fad_extractor,
        clap_model,
        device,
    )
    typer.echo({"job_id": job.job_id, "mode": mode, "metrics": metric_list})


@app.command("evaluation-summary")
def evaluation_summary_command(evaluation_id: str) -> None:
    from hybrid_music_engine.evaluation.pipeline import load_evaluation_report
    typer.echo(load_evaluation_report(config, evaluation_id))


@app.command("evaluation-export")
def evaluation_export_command(evaluation_id: str) -> None:
    from hybrid_music_engine.evaluation.pipeline import evaluation_files
    typer.echo(evaluation_files(config, evaluation_id))


@app.command("classifier-train")
def classifier_train_command(
    real_audio_root: Annotated[Path, typer.Option("--real-root")],
    labels: Annotated[str, typer.Option("--labels")] = "",
    output_name: Annotated[str, typer.Option("--name")] = "audio_classifier",
    max_files_per_class: Annotated[int | None, typer.Option("--max-files-per-class")] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
) -> None:
    from hybrid_music_engine.jobs.dispatcher import dispatch_classifier_train
    parsed_labels = [item.strip() for item in labels.split(",") if item.strip()] or None
    job = create_job(config, kind="classifier-train")
    mode = dispatch_classifier_train(
        job.job_id,
        str(real_audio_root),
        parsed_labels,
        output_name,
        max_files_per_class,
        temperature,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


@app.command("classifier-predict")
def classifier_predict_command(
    model_path: Annotated[Path | None, typer.Option("--model")] = None,
    audio_root: Annotated[Path | None, typer.Option("--audio-root")] = None,
    audio_paths: Annotated[str, typer.Option("--audio")] = "",
) -> None:
    from hybrid_music_engine.jobs.dispatcher import dispatch_classifier_predict
    parsed_paths = [item.strip() for item in audio_paths.split(",") if item.strip()]
    job = create_job(config, kind="classifier-predict")
    mode = dispatch_classifier_predict(
        job.job_id,
        str(model_path) if model_path else None,
        parsed_paths,
        str(audio_root) if audio_root else None,
    )
    typer.echo({"job_id": job.job_id, "mode": mode})


def _parse_evaluation_distribution(value: str, real_audio_root: Path | None, target_total: int) -> dict[str, int] | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    from hybrid_music_engine.evaluation.pipeline import evaluation_availability
    if normalized == "max":
        return evaluation_availability(config, real_audio_root=real_audio_root, target_total=target_total)["max_distribution"]
    parsed: dict[str, int] = {}
    for item in value.split(","):
        raw = item.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise typer.BadParameter("Usa auto, max o genre=numero separado por comas.")
        genre, amount = raw.split("=", 1)
        parsed[genre.strip()] = int(amount.strip())
    if not parsed:
        raise typer.BadParameter("Distribución vacía.")
    return parsed
