from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from hybrid_music_engine.api.schemas import (
    BuildGenreCatalogRequest,
    AugmentMidiRequest,
    CleanMidiDatasetRequest,
    BlendEmbeddingsRequest,
    BlendWeightedEmbeddingsRequest,
    CompareFusionsRequest,
    CreateProjectRequest,
    DownloadJamendoRequest,
    EncodeProjectRequest,
    EncodeGenreEmbeddingsRequest,
    EncodeTokenVAERequest,
    ExportInputTokensRequest,
    ExportOutputTokensRequest,
    ExtractFeaturesRequest,
    GenerationPlanRequest,
    GeneratePretrainedRequest,
    GenerateRankedRequest,
    GenerateTokensRequest,
    ImportAudioRequest,
    JobCreatedResponse,
    MidiMetricsRequest,
    PrepareJamendoClipsRequest,
    ProcessJamendoClipsRequest,
    RenderLayersRequest,
    RenderMidiRequest,
    ProjectCreatedResponse,
    SelectJamendoCatalogRequest,
    SeparateStemsRequest,
    TokenizeCatalogRequest,
    TrainTokenModelRequest,
    TrainTokenVAERequest,
    TrainVAERequest,
    TranscribeDrumsRequest,
    TranscribeMelodicRequest,
)
from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.presets import presets_payload
from hybrid_music_engine.datasets.cleaning import clean_midi_dataset
from hybrid_music_engine.datasets.genre_catalog import build_genre_catalog
from hybrid_music_engine.datasets.jamendo import list_jamendo_catalogs, select_jamendo_catalog_entries
from hybrid_music_engine.jobs.dispatcher import (
    dispatch_blend_embeddings,
    dispatch_blend_weighted_embeddings,
    dispatch_compare_fusions,
    dispatch_augment_midi,
    dispatch_download_jamendo,
    dispatch_encode_project,
    dispatch_encode_genre_embeddings,
    dispatch_encode_token_vae,
    dispatch_extract_features,
    dispatch_generate_tokens,
    dispatch_generate_ranked,
    dispatch_import_audio,
    dispatch_midi_metrics,
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
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
from hybrid_music_engine.storage.job_store import create_job, list_jobs, load_job
from hybrid_music_engine.storage.catalog import (
    list_blends,
    list_fusion_comparisons,
    list_generations,
    list_renders,
    list_rankings,
    list_token_vae_assets,
    list_token_manifests,
    list_token_models,
)
from hybrid_music_engine.storage.manifest import create_project, list_projects, load_manifest
from hybrid_music_engine.tokens.midi_tokenizer import (
    create_generation_plan,
    export_token_manifest_to_zip,
    export_output_tokens_to_zip,
    tokenize_catalog_to_zip,
)


from hybrid_music_engine.reward_model.api_router import router as reward_router


app = FastAPI(title="Motor de Transformacion Musical", version="0.1.0")
config = EngineConfig.from_env()
config.ensure_directories()

# Origenes permitidos: localhost (dev) + cualquier sitio estatico desplegado
# (Netlify/Vercel u otro dominio). Como el frontend no usa cookies, se desactiva
# allow_credentials para poder permitir cualquier origen de forma segura.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reward_router, prefix="/api/reward", tags=["reward"])


@app.get("/")
def root() -> dict:
    return {
        "ok": True,
        "service": "Motor de Transformacion Musical",
        "health": "/api/health",
        "docs": "/docs",
        "frontend": "http://127.0.0.1:5173",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "project_root": str(config.project_root),
        "projects_dir": str(config.projects_dir),
        "jobs_dir": str(config.jobs_dir),
        "sample_rate": config.default_sample_rate,
        "channels": config.default_channels,
        "celery_broker_url": config.celery_broker_url,
        "job_backend": config.job_backend,
        "require_celery": config.require_celery,
    }


@app.post("/api/projects", response_model=ProjectCreatedResponse)
def create_project_endpoint(request: CreateProjectRequest) -> dict:
    manifest = create_project(config, request.name)
    return {
        "project_id": manifest.project_id,
        "name": manifest.name,
        "manifest": asdict(manifest),
    }


@app.get("/api/projects")
def projects() -> dict:
    return {"projects": list_projects(config)}


@app.get("/api/projects/{project_id}")
def project_detail(project_id: str) -> dict:
    try:
        return asdict(load_manifest(config, project_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/import-audio", response_model=JobCreatedResponse)
def import_audio_job(request: ImportAudioRequest) -> dict:
    try:
        load_manifest(config, request.project_id)
        job = create_job(config, kind="import-audio", project_id=request.project_id)
        mode = dispatch_import_audio(job.job_id, request.project_id, str(request.source_path))
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/separate-stems", response_model=JobCreatedResponse)
def separate_stems_job(request: SeparateStemsRequest) -> dict:
    try:
        load_manifest(config, request.project_id)
        job = create_job(config, kind="separate-stems", project_id=request.project_id)
        mode = dispatch_separate_stems(
            job.job_id,
            request.project_id,
            str(request.audio_path) if request.audio_path else None,
            request.model_name,
            request.device,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/transcribe-melodic", response_model=JobCreatedResponse)
def transcribe_melodic_job(request: TranscribeMelodicRequest) -> dict:
    try:
        load_manifest(config, request.project_id)
        job = create_job(config, kind="transcribe-melodic", project_id=request.project_id)
        mode = dispatch_transcribe_melodic(
            job.job_id,
            request.project_id,
            request.stems,
            str(request.audio_path) if request.audio_path else None,
            request.minimum_note_length,
            request.onset_threshold,
            request.frame_threshold,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/transcribe-drums", response_model=JobCreatedResponse)
def transcribe_drums_job(request: TranscribeDrumsRequest) -> dict:
    try:
        load_manifest(config, request.project_id)
        job = create_job(config, kind="transcribe-drums", project_id=request.project_id)
        mode = dispatch_transcribe_drums(
            job.job_id,
            request.project_id,
            str(request.audio_path) if request.audio_path else None,
            request.bpm,
            request.onset_delta,
            request.onset_wait,
            request.note_length,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/extract-features", response_model=JobCreatedResponse)
def extract_features_job(request: ExtractFeaturesRequest) -> dict:
    try:
        load_manifest(config, request.project_id)
        job = create_job(config, kind="extract-features", project_id=request.project_id)
        mode = dispatch_extract_features(
            job.job_id,
            request.project_id,
            request.include_audio,
            request.include_midis,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/train-vae", response_model=JobCreatedResponse)
def train_vae_job(request: TrainVAERequest) -> dict:
    try:
        job = create_job(config, kind="train-vae")
        mode = dispatch_train_vae(
            job.job_id,
            request.latent_dim,
            request.hidden_dim,
            request.epochs,
            request.learning_rate,
            request.beta,
            request.seed,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/encode-project", response_model=JobCreatedResponse)
def encode_project_job(request: EncodeProjectRequest) -> dict:
    try:
        load_manifest(config, request.project_id)
        job = create_job(config, kind="encode-project", project_id=request.project_id)
        mode = dispatch_encode_project(
            job.job_id,
            request.project_id,
            str(request.model_path) if request.model_path else None,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/augment-midi", response_model=JobCreatedResponse)
def augment_midi_job(request: AugmentMidiRequest) -> dict:
    job = create_job(config, kind="augment-midi")
    mode = dispatch_augment_midi(
        job.job_id,
        str(request.catalog_path) if request.catalog_path else None,
        str(request.source_dir) if request.source_dir else None,
        request.output_name,
        request.transpose_steps,
        request.velocity_jitter,
        request.timing_jitter_ticks,
        request.quantize_step_ticks,
        request.tempo_scale,
        request.seed,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/train-token-vae", response_model=JobCreatedResponse)
def train_token_vae_job(request: TrainTokenVAERequest) -> dict:
    job = create_job(config, kind="train-token-vae")
    mode = dispatch_train_token_vae(
        job.job_id,
        str(request.token_manifest_path),
        request.latent_dim,
        request.hidden_dim,
        request.epochs,
        request.learning_rate,
        request.beta,
        request.seed,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/encode-token-vae", response_model=JobCreatedResponse)
def encode_token_vae_job(request: EncodeTokenVAERequest) -> dict:
    job = create_job(config, kind="encode-token-vae")
    mode = dispatch_encode_token_vae(
        job.job_id,
        str(request.token_source_path),
        str(request.model_path) if request.model_path else None,
        request.output_name,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/encode-genre-embeddings", response_model=JobCreatedResponse)
def encode_genre_embeddings_job(request: EncodeGenreEmbeddingsRequest) -> dict:
    job = create_job(config, kind="encode-genre-embeddings")
    mode = dispatch_encode_genre_embeddings(
        job.job_id,
        str(request.token_manifest_path),
        str(request.model_path) if request.model_path else None,
        request.output_name,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.get("/api/embeddings/vae")
def vae_status() -> dict:
    metadata_path = config.data_dir / "embeddings" / "feature_vae" / "metadata.json"
    model_path = config.data_dir / "embeddings" / "feature_vae" / "model.pt"
    if not metadata_path.exists() or not model_path.exists():
        return {
            "available": False,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metadata": None,
        }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo leer metadata VAE: {exc}") from exc
    return {
        "available": True,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


@app.get("/api/resources")
def resources() -> dict:
    return {
        "token_manifests": list_token_manifests(config),
        "token_models": list_token_models(config),
        "generations": list_generations(config),
        "rankings": list_rankings(config),
        "renders": list_renders(config),
        "blends": list_blends(config),
        "fusion_comparisons": list_fusion_comparisons(config),
        "token_vae": list_token_vae_assets(config),
        "jamendo_catalogs": list_jamendo_catalogs(config),
    }


@app.get("/api/presets")
def presets() -> dict:
    return presets_payload()


@app.get("/api/tokens/manifests")
def token_manifests() -> dict:
    return {"token_manifests": list_token_manifests(config)}


@app.get("/api/models/tokens")
def token_models() -> dict:
    return {"token_models": list_token_models(config)}


@app.get("/api/generated")
def generated() -> dict:
    return {"generations": list_generations(config)}


@app.get("/api/rankings")
def rankings() -> dict:
    return {"rankings": list_rankings(config)}


@app.get("/api/renders")
def renders() -> dict:
    return {"renders": list_renders(config)}


@app.get("/api/datasets/jamendo/catalogs")
def jamendo_catalogs() -> dict:
    return {"catalogs": list_jamendo_catalogs(config)}


@app.post("/api/datasets/jamendo/select")
def select_jamendo_catalog(request: SelectJamendoCatalogRequest) -> dict:
    try:
        return select_jamendo_catalog_entries(
            config,
            catalog_path=request.catalog_path,
            genres=request.genres,
            max_tracks_per_genre=request.max_tracks_per_genre,
            output_name=request.output_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/blends")
def blends() -> dict:
    return {"blends": list_blends(config)}


@app.post("/api/datasets/genre-catalog")
def build_genre_catalog_endpoint(request: BuildGenreCatalogRequest) -> dict:
    try:
        return build_genre_catalog(
            config,
            source_dir=request.source_dir,
            genres=request.genres,
            clips_per_genre=request.clips_per_genre,
            max_duration_seconds=request.max_duration_seconds,
            catalog_name=request.catalog_name,
            source_label=request.source_label,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/clean-midi")
def clean_midi_dataset_endpoint(request: CleanMidiDatasetRequest) -> dict:
    try:
        return clean_midi_dataset(
            config,
            source_dir=request.source_dir,
            output_name=request.output_name,
            min_duration_seconds=request.min_duration_seconds,
            max_duration_seconds=request.max_duration_seconds,
            min_notes=request.min_notes,
            min_quality_score=request.min_quality_score,
            deduplicate=request.deduplicate,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/download-jamendo", response_model=JobCreatedResponse)
def download_jamendo_job(request: DownloadJamendoRequest) -> dict:
    try:
        job = create_job(config, kind="download-jamendo")
        mode = dispatch_download_jamendo(
            job.job_id,
            request.genre_tags,
            request.catalog_name,
            request.tracks_per_page,
            request.max_tracks_per_genre,
            request.download_audio,
            request.client_id,
            request.source,
            request.concurrent_downloads,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/prepare-jamendo-clips", response_model=JobCreatedResponse)
def prepare_jamendo_clips_job(request: PrepareJamendoClipsRequest) -> dict:
    try:
        job = create_job(config, kind="prepare-jamendo-clips")
        mode = dispatch_prepare_jamendo_clips(
            job.job_id,
            str(request.catalog_path),
            request.clip_duration_seconds,
            request.hop_duration_seconds,
            request.max_clips_per_track,
            request.min_clip_seconds,
            request.sample_rate,
            request.mono,
        )
        return {"job_id": job.job_id, "mode": mode}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/process-jamendo-clips", response_model=JobCreatedResponse)
def process_jamendo_clips_job(request: ProcessJamendoClipsRequest) -> dict:
    job = create_job(config, kind="process-jamendo-clips")
    mode = dispatch_process_jamendo_clips(
        job.job_id,
        str(request.clips_catalog_path),
        request.max_clips,
        request.run_stems,
        request.run_melodic,
        request.run_drums,
        request.run_features,
        request.run_tokens,
        request.continue_on_error,
        request.processing_mode,
        request.midi_cleanup,
        request.quantize_grid,
        request.strict_demucs,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/train-token-model", response_model=JobCreatedResponse)
def train_token_model_job(request: TrainTokenModelRequest) -> dict:
    job = create_job(config, kind="train-token-model")
    mode = dispatch_train_token_model(
        job.job_id,
        str(request.token_manifest_path),
        request.model_name,
        request.order,
        request.model_type,
        request.sequence_length,
        request.epochs,
        request.batch_size,
        request.embedding_dim,
        request.num_layers,
        request.num_heads,
    )
    return {"job_id": job.job_id, "mode": mode}


_VALID_PRETRAINED = {"musicgen-small", "musicgen-medium", "audioldm2", "stable-audio-open"}


@app.post("/api/jobs/generate-pretrained", response_model=JobCreatedResponse)
def generate_pretrained_job(request: GeneratePretrainedRequest) -> dict:
    if request.model_name not in _VALID_PRETRAINED:
        raise HTTPException(status_code=422, detail=f"model_name debe ser uno de {sorted(_VALID_PRETRAINED)}")
    job = create_job(config, kind="generate-pretrained")
    from hybrid_music_engine.jobs.dispatcher import dispatch_generate_pretrained
    mode = dispatch_generate_pretrained(job.job_id, request.model_name)
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/generate-tokens", response_model=JobCreatedResponse)
def generate_tokens_job(request: GenerateTokensRequest) -> dict:
    job = create_job(config, kind="generate-tokens")
    mode = dispatch_generate_tokens(
        job.job_id,
        str(request.model_path),
        request.duration_seconds,
        request.output_name,
        request.seed,
        request.max_tokens,
        request.temperature,
        request.top_k,
        request.top_p,
        request.condition_genre,
        request.feature_tokens,
        str(request.embedding_path) if request.embedding_path else None,
        str(request.token_vae_embedding_path) if request.token_vae_embedding_path else None,
        request.export_layers,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/generate-ranked", response_model=JobCreatedResponse)
def generate_ranked_job(request: GenerateRankedRequest) -> dict:
    job = create_job(config, kind="generate-ranked")
    mode = dispatch_generate_ranked(
        job.job_id,
        str(request.model_path),
        request.duration_seconds,
        request.output_name,
        request.candidates,
        request.seed,
        request.max_tokens,
        request.temperature,
        request.top_k,
        request.top_p,
        request.condition_genre,
        request.feature_tokens,
        str(request.embedding_path) if request.embedding_path else None,
        str(request.token_vae_embedding_path) if request.token_vae_embedding_path else None,
        request.export_layers,
        request.render_best,
        request.render_engine,
        str(request.soundfont_path) if request.soundfont_path else None,
        request.export_mp3,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/blend-embeddings", response_model=JobCreatedResponse)
def blend_embeddings_job(request: BlendEmbeddingsRequest) -> dict:
    job = create_job(config, kind="blend-embeddings")
    mode = dispatch_blend_embeddings(
        job.job_id,
        str(request.embedding_a_path),
        str(request.embedding_b_path),
        request.alpha,
        request.output_name,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/blend-weighted-embeddings", response_model=JobCreatedResponse)
def blend_weighted_embeddings_job(request: BlendWeightedEmbeddingsRequest) -> dict:
    job = create_job(config, kind="blend-weighted-embeddings")
    mode = dispatch_blend_weighted_embeddings(
        job.job_id,
        [
            {"path": str(item.path), "weight": item.weight, "label": item.label}
            for item in request.embeddings
        ],
        request.output_name,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/compare-fusions", response_model=JobCreatedResponse)
def compare_fusions_job(request: CompareFusionsRequest) -> dict:
    job = create_job(config, kind="compare-fusions")
    mode = dispatch_compare_fusions(
        job.job_id,
        str(request.model_path),
        [
            {"embedding_path": str(item.embedding_path), "label": item.label}
            for item in request.fusion_embeddings
        ],
        request.duration_seconds,
        request.output_name,
        request.candidates_per_fusion,
        request.seed,
        request.max_tokens,
        request.temperature,
        request.top_k,
        request.top_p,
        request.feature_tokens,
        request.export_layers,
        request.render_best,
        request.render_engine,
        str(request.soundfont_path) if request.soundfont_path else None,
        request.export_mp3,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/render-midi", response_model=JobCreatedResponse)
def render_midi_job(request: RenderMidiRequest) -> dict:
    job = create_job(config, kind="render-midi")
    mode = dispatch_render_midi(
        job.job_id,
        str(request.midi_path),
        request.output_name,
        request.engine,
        str(request.soundfont_path) if request.soundfont_path else None,
        request.sample_rate,
        request.export_mp3,
        request.pedalboard_preset,
        [str(path) for path in request.plugin_paths],
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/render-layers", response_model=JobCreatedResponse)
def render_layers_job(request: RenderLayersRequest) -> dict:
    job = create_job(config, kind="render-layers")
    mode = dispatch_render_layers(
        job.job_id,
        str(request.generation_path),
        request.output_name,
        request.engine,
        str(request.soundfont_path) if request.soundfont_path else None,
        request.sample_rate,
        request.export_mp3,
        request.pedalboard_preset,
        [str(path) for path in request.plugin_paths],
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/midi-metrics", response_model=JobCreatedResponse)
def midi_metrics_job(request: MidiMetricsRequest) -> dict:
    job = create_job(config, kind="midi-metrics")
    mode = dispatch_midi_metrics(job.job_id, str(request.midi_path))
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/metrics/midi")
def midi_metrics_endpoint(request: MidiMetricsRequest) -> dict:
    try:
        return {"metrics": analyze_midi_quality(request.midi_path)}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/metrics/embedding-projection")
def metrics_embedding_projection(run_path: str, seed: int = 42, max_per_genre: int = 80) -> dict:
    try:
        from hybrid_music_engine.new_metrics.embedding_projection import (
            embedding_projection_from_run,
        )

        return embedding_projection_from_run(
            config,
            Path(run_path),
            seed=seed,
            max_files_per_genre=max_per_genre,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/metrics/token-tsne")
def metrics_token_tsne(
    t5_dir: str | None = None,
    encodec_dir: str | None = None,
    pool_text: bool = False,
    max_points: int = 400,
    knn: int = 5,
    draw_edges: bool = True,
    seed: int = 42,
) -> dict:
    try:
        from hybrid_music_engine.new_metrics.token_tsne import token_tsne_projection

        return token_tsne_projection(
            config,
            t5_dir=t5_dir,
            encodec_dir=encodec_dir,
            pool_text=pool_text,
            max_points=max_points,
            knn=knn,
            draw_edges=draw_edges,
            seed=seed,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/metrics/benchmark")
def metrics_benchmark(set_level: str | None = None, clip_level: str | None = None) -> dict:
    try:
        from hybrid_music_engine.new_metrics.benchmark_analysis import benchmark_payload

        results_dir = config.project_root / "results"
        set_csv = Path(set_level) if set_level else results_dir / "set_level_metrics.csv"
        clip_csv = Path(clip_level) if clip_level else results_dir / "clip_level_metrics.csv"
        return benchmark_payload(set_csv, clip_csv)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tokens/input")
def tokenize_catalog_endpoint(request: TokenizeCatalogRequest) -> dict:
    try:
        payload = tokenize_catalog_to_zip(
            config,
            catalog_path=request.catalog_path,
            token_set_name=request.token_set_name,
        )
        return _with_download_url(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tokens/input/export")
def export_input_tokens_endpoint(request: ExportInputTokensRequest) -> dict:
    try:
        payload = export_token_manifest_to_zip(
            config,
            token_manifest_path=request.token_manifest_path,
            export_name=request.export_name,
        )
        return _with_download_url(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tokens/output")
def export_output_tokens_endpoint(request: ExportOutputTokensRequest) -> dict:
    try:
        payload = export_output_tokens_to_zip(
            config,
            source_dir=request.source_dir,
            export_name=request.export_name,
            duration_seconds=request.duration_seconds,
        )
        return _with_download_url(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/generation/plan")
def generation_plan_endpoint(request: GenerationPlanRequest) -> dict:
    try:
        return create_generation_plan(
            config,
            project_id=request.project_id,
            duration_seconds=request.duration_seconds,
            output_name=request.output_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/artifacts/{filename}")
def download_artifact(filename: str) -> FileResponse:
    path = (config.artifacts_dir / filename).resolve()
    artifacts_root = config.artifacts_dir.resolve()
    if not path.is_relative_to(artifacts_root) or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")
    return FileResponse(path, filename=path.name)


@app.get("/api/files")
def download_data_file(path: str = Query(...)) -> FileResponse:
    requested = Path(path).expanduser().resolve()
    allowed_roots = [config.data_dir.resolve(), config.artifacts_dir.resolve()]
    if (
        not any(requested.is_relative_to(root) for root in allowed_roots)
        or not requested.exists()
        or not requested.is_file()
    ):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return FileResponse(requested, filename=requested.name)


@app.get("/api/jobs")
def jobs() -> dict:
    return {"jobs": list_jobs(config)}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    try:
        return asdict(load_job(config, job_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _with_download_url(payload: dict) -> dict:
    zip_path = payload.get("zip_path")
    if zip_path:
        payload["download_url"] = f"/api/artifacts/{Path(str(zip_path)).name}"
    return payload



# --- Evaluación formal y métricas reales ---
from hybrid_music_engine.api.schemas import (  # noqa: E402
    ClassifierPredictRequest,
    ClassifierTrainRequest,
    EvaluationFromResultsRequest,
    EvaluationGenerateBatchRequest,
    EvaluationRunRequest,
)
from hybrid_music_engine.evaluation.pipeline import (  # noqa: E402
    evaluation_availability,
    evaluation_files,
    evaluation_generated_sources,
    list_evaluations,
    load_evaluation,
    load_evaluation_report,
)


@app.get("/api/evaluations/availability")
def evaluations_availability(real_audio_root: str | None = None, target_total: int = 100) -> dict:
    try:
        return evaluation_availability(
            config,
            real_audio_root=Path(real_audio_root) if real_audio_root else None,
            target_total=target_total,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/evaluations/generated-sources")
def evaluations_generated_sources() -> dict:
    try:
        return evaluation_generated_sources(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/evaluations")
def evaluations() -> dict:
    return {"evaluations": list_evaluations(config)}


@app.get("/api/evaluations/{evaluation_id}")
def evaluation_detail(evaluation_id: str) -> dict:
    try:
        return load_evaluation(config, evaluation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/evaluations/{evaluation_id}/report")
def evaluation_report(evaluation_id: str) -> dict:
    try:
        return load_evaluation_report(config, evaluation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/evaluations/{evaluation_id}/files")
def evaluation_file_list(evaluation_id: str) -> dict:
    try:
        return evaluation_files(config, evaluation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/evaluation/generate-batch", response_model=JobCreatedResponse)
def evaluation_generate_batch_job(request: EvaluationGenerateBatchRequest) -> dict:
    from hybrid_music_engine.jobs.dispatcher import dispatch_evaluation_generate_batch

    job = create_job(config, kind="evaluation-generate-batch")
    mode = dispatch_evaluation_generate_batch(
        job.job_id,
        str(request.model_path),
        request.distribution,
        str(request.real_audio_root) if request.real_audio_root else None,
        request.duration_seconds,
        request.output_name,
        request.seed,
        request.max_tokens,
        request.temperature,
        request.top_k,
        request.top_p,
        request.export_layers,
        request.render_audio,
        request.render_engine,
        request.export_mp3,
        request.target_total,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/evaluation/run", response_model=JobCreatedResponse)
def evaluation_run_job(request: EvaluationRunRequest) -> dict:
    from hybrid_music_engine.jobs.dispatcher import dispatch_evaluation_run

    job = create_job(config, kind="evaluation-run")
    mode = dispatch_evaluation_run(
        job.job_id,
        request.evaluation_id,
        str(request.generated_root) if request.generated_root else None,
        str(request.real_root) if request.real_root else None,
        str(request.prompts_path) if request.prompts_path else None,
        str(request.classifier_path) if request.classifier_path else None,
        request.train_classifier_if_missing,
        request.metrics,
        request.fad_extractor,
        request.clap_model,
        request.device,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/evaluation/from-results", response_model=JobCreatedResponse)
def evaluation_from_results_job(request: EvaluationFromResultsRequest) -> dict:
    from hybrid_music_engine.jobs.dispatcher import dispatch_evaluation_from_results

    job = create_job(config, kind="evaluation-from-results")
    metrics = list(request.metrics)
    if request.include_clap and "clap" not in metrics:
        metrics.append("clap")
    selections = [selection.model_dump() for selection in request.selections]
    genre_selections = {
        genre: [selection.model_dump() for selection in rows]
        for genre, rows in request.genre_selections.items()
    }
    if not selections and request.source_id:
        selections = [
            {
                "source_type": request.source_type or "ranking",
                "source_id": request.source_id,
                "limit": request.limit or 1,
                "candidate_ids": [],
            }
        ]
    mode = dispatch_evaluation_from_results(
        job.job_id,
        selections,
        genre_selections,
        request.target_per_genre,
        request.pairing_strategy,
        str(request.real_audio_root) if request.real_audio_root else None,
        request.output_name,
        metrics,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/classifier/train", response_model=JobCreatedResponse)
def classifier_train_job(request: ClassifierTrainRequest) -> dict:
    from hybrid_music_engine.jobs.dispatcher import dispatch_classifier_train

    job = create_job(config, kind="classifier-train")
    mode = dispatch_classifier_train(
        job.job_id,
        str(request.real_audio_root),
        request.labels,
        request.output_name,
        request.max_files_per_class,
        request.temperature,
    )
    return {"job_id": job.job_id, "mode": mode}


@app.post("/api/jobs/classifier/predict", response_model=JobCreatedResponse)
def classifier_predict_job(request: ClassifierPredictRequest) -> dict:
    from hybrid_music_engine.jobs.dispatcher import dispatch_classifier_predict

    job = create_job(config, kind="classifier-predict")
    mode = dispatch_classifier_predict(
        job.job_id,
        str(request.model_path) if request.model_path else None,
        [str(path) for path in request.audio_paths],
        str(request.audio_root) if request.audio_root else None,
    )
    return {"job_id": job.job_id, "mode": mode}
