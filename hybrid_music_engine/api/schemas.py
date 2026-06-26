from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CreateProjectRequest(BaseModel):
    name: str = Field(default="untitled")


class ImportAudioRequest(BaseModel):
    project_id: str
    source_path: Path


class SeparateStemsRequest(BaseModel):
    project_id: str
    audio_path: Path | None = None
    model_name: str = "htdemucs"
    device: str = "auto"


class TranscribeMelodicRequest(BaseModel):
    project_id: str
    stems: list[str] | None = Field(default_factory=lambda: ["bass", "vocals", "other"])
    audio_path: Path | None = None
    minimum_note_length: float | None = None
    onset_threshold: float | None = None
    frame_threshold: float | None = None


class TranscribeDrumsRequest(BaseModel):
    project_id: str
    audio_path: Path | None = None
    bpm: float | None = None
    onset_delta: float = 0.07
    onset_wait: float = 0.03
    note_length: float = 0.08


class ExtractFeaturesRequest(BaseModel):
    project_id: str
    include_audio: bool = True
    include_midis: bool = True


class TrainVAERequest(BaseModel):
    latent_dim: int = 32
    hidden_dim: int = 128
    epochs: int = 200
    learning_rate: float = 1e-3
    beta: float = 0.001
    seed: int = 42


class AugmentMidiRequest(BaseModel):
    catalog_path: Path | None = None
    source_dir: Path | None = None
    output_name: str = "augmented_midis"
    transpose_steps: list[int] = Field(default_factory=lambda: [-2, 0, 2])
    velocity_jitter: int = 8
    timing_jitter_ticks: int = 12
    quantize_step_ticks: int | None = None
    tempo_scale: float = 1.0
    seed: int = 42


class TrainTokenVAERequest(BaseModel):
    token_manifest_path: Path
    latent_dim: int = 32
    hidden_dim: int = 128
    epochs: int = 80
    learning_rate: float = 1e-3
    beta: float = 0.001
    seed: int = 42


class EncodeTokenVAERequest(BaseModel):
    token_source_path: Path
    model_path: Path | None = None
    output_name: str = "token_embedding"


class EncodeGenreEmbeddingsRequest(BaseModel):
    token_manifest_path: Path
    model_path: Path | None = None
    output_name: str = "genre_embeddings"


class EncodeProjectRequest(BaseModel):
    project_id: str
    model_path: Path | None = None


class BuildGenreCatalogRequest(BaseModel):
    source_dir: Path
    genres: list[str]
    clips_per_genre: int = 200
    max_duration_seconds: float = 10.0
    catalog_name: str = "genre_catalog"
    source_label: str = "source_3"


class CleanMidiDatasetRequest(BaseModel):
    source_dir: Path
    output_name: str = "clean_midis"
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 240.0
    min_notes: int = 4
    min_quality_score: float = 0.05
    deduplicate: bool = True


class TokenizeCatalogRequest(BaseModel):
    catalog_path: Path
    token_set_name: str = "input_tokens"


class ExportInputTokensRequest(BaseModel):
    token_manifest_path: Path
    export_name: str = "input_tokens"


class ExportOutputTokensRequest(BaseModel):
    source_dir: Path
    export_name: str = "mixed_output_tokens"
    duration_seconds: float | None = None


class GenerationPlanRequest(BaseModel):
    project_id: str | None = None
    duration_seconds: float
    output_name: str = "generated_track"


class DownloadJamendoRequest(BaseModel):
    genre_tags: dict[str, list[str]] | None = None
    catalog_name: str = "mtg_jamendo"
    tracks_per_page: int = 200
    max_tracks_per_genre: int | None = 500
    download_audio: bool = True
    client_id: str = "b6747d04"
    source: str = "mtg-cdn"
    concurrent_downloads: int = 16


class PrepareJamendoClipsRequest(BaseModel):
    catalog_path: Path
    clip_duration_seconds: float = 20.0
    hop_duration_seconds: float | None = None
    max_clips_per_track: int | None = None
    min_clip_seconds: float = 5.0
    sample_rate: int | None = None
    mono: bool = True


class ProcessJamendoClipsRequest(BaseModel):
    clips_catalog_path: Path
    max_clips: int | None = None
    run_stems: bool = False
    run_melodic: bool = True
    run_drums: bool = True
    run_features: bool = True
    run_tokens: bool = True
    continue_on_error: bool = True
    processing_mode: Literal["quick", "token_vae_demucs"] = "quick"
    midi_cleanup: bool = False
    quantize_grid: Literal["1/8", "1/16", "1/32"] = "1/16"
    strict_demucs: bool = False


class SelectJamendoCatalogRequest(BaseModel):
    catalog_path: Path
    genres: list[str]
    max_tracks_per_genre: int = 100
    output_name: str = "selected_jamendo"


class TrainTokenModelRequest(BaseModel):
    token_manifest_path: Path
    model_name: str = "token_markov"
    order: int = 2
    model_type: str = "markov"
    sequence_length: int = 128
    epochs: int = 8
    batch_size: int = 16
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4


class GenerateTokensRequest(BaseModel):
    model_path: Path
    duration_seconds: float
    output_name: str = "generated"
    seed: int | None = None
    max_tokens: int | None = None
    temperature: float = 0.9
    top_k: int | None = 50
    top_p: float | None = 0.95
    condition_genre: str | None = None
    feature_tokens: list[str] = Field(default_factory=list)
    embedding_path: Path | None = None
    token_vae_embedding_path: Path | None = None
    export_layers: bool = True


class GenerateRankedRequest(BaseModel):
    model_path: Path
    duration_seconds: float
    output_name: str = "ranked_generation"
    candidates: int = 6
    seed: int | None = None
    max_tokens: int | None = None
    temperature: float = 0.9
    top_k: int | None = 50
    top_p: float | None = 0.95
    condition_genre: str | None = None
    feature_tokens: list[str] = Field(default_factory=list)
    embedding_path: Path | None = None
    token_vae_embedding_path: Path | None = None
    export_layers: bool = True
    render_best: bool = False
    render_engine: str = "auto"
    soundfont_path: Path | None = None
    export_mp3: bool = False


class RenderLayersRequest(BaseModel):
    generation_path: Path
    output_name: str = "layer_render"
    engine: str = "auto"
    soundfont_path: Path | None = None
    sample_rate: int = 44100
    export_mp3: bool = False
    pedalboard_preset: str = "master"
    plugin_paths: list[Path] = Field(default_factory=list)


class FusionComparisonItem(BaseModel):
    embedding_path: Path
    label: str | None = None


class CompareFusionsRequest(BaseModel):
    model_path: Path
    fusion_embeddings: list[FusionComparisonItem]
    duration_seconds: float = 30
    output_name: str = "fusion_comparison"
    candidates_per_fusion: int = 3
    seed: int | None = 42
    max_tokens: int | None = 1200
    temperature: float = 0.84
    top_k: int | None = 56
    top_p: float | None = 0.92
    feature_tokens: list[str] = Field(default_factory=list)
    export_layers: bool = True
    render_best: bool = False
    render_engine: str = "auto"
    soundfont_path: Path | None = None
    export_mp3: bool = False


class MidiMetricsRequest(BaseModel):
    midi_path: Path


class BlendEmbeddingsRequest(BaseModel):
    embedding_a_path: Path
    embedding_b_path: Path
    alpha: float = 0.5
    output_name: str = "latent_blend"


class WeightedEmbeddingInput(BaseModel):
    path: Path
    weight: float
    label: str | None = None


class BlendWeightedEmbeddingsRequest(BaseModel):
    embeddings: list[WeightedEmbeddingInput]
    output_name: str = "genre_fusion"


class RenderMidiRequest(BaseModel):
    midi_path: Path
    output_name: str = "preview"
    engine: str = "auto"
    soundfont_path: Path | None = None
    sample_rate: int = 44100
    export_mp3: bool = False
    pedalboard_preset: str = "master"
    plugin_paths: list[Path] = Field(default_factory=list)


class GeneratePretrainedRequest(BaseModel):
    model_name: str  # musicgen-small | musicgen-medium | audioldm2 | stable-audio-open


class JobCreatedResponse(BaseModel):
    job_id: str
    mode: str


class ProjectCreatedResponse(BaseModel):
    project_id: str
    name: str
    manifest: dict


class EvaluationGenerateBatchRequest(BaseModel):
    model_path: Path
    distribution: dict[str, int] | None = None
    real_audio_root: Path | None = None
    duration_seconds: float = 30.0
    output_name: str = "evaluation_batch"
    seed: int | None = 42
    max_tokens: int | None = None
    temperature: float = 0.9
    top_k: int | None = 50
    top_p: float | None = 0.95
    export_layers: bool = True
    render_audio: bool = True
    render_engine: str = "auto"
    export_mp3: bool = True
    target_total: int = 100


class EvaluationRunRequest(BaseModel):
    evaluation_id: str | None = None
    generated_root: Path | None = None
    real_root: Path | None = None
    prompts_path: Path | None = None
    classifier_path: Path | None = None
    train_classifier_if_missing: bool = True
    metrics: list[str] = Field(default_factory=lambda: ["fad", "kad", "kld", "tempo", "midi"])
    fad_extractor: str = "mel"
    clap_model: str = "clap"
    device: str = "cpu"


class EvaluationResultSelection(BaseModel):
    source_type: str = "ranking"
    source_id: str
    limit: int = 1
    candidate_ids: list[str] = Field(default_factory=list)
    genre: str | None = None


class EvaluationGenreSelection(BaseModel):
    source_type: str = "ranking"
    source_id: str
    limit: int = 1
    candidate_ids: list[str] = Field(default_factory=list)


class EvaluationFromResultsRequest(BaseModel):
    selections: list[EvaluationResultSelection] = Field(default_factory=list)
    genre_selections: dict[str, list[EvaluationGenreSelection]] = Field(default_factory=dict)
    target_per_genre: int = 20
    pairing_strategy: str = "same_genre_round_robin"
    source_type: str | None = None
    source_id: str | None = None
    limit: int | None = None
    real_audio_root: Path | None = None
    output_name: str = "generated_results"
    metrics: list[str] = Field(default_factory=lambda: ["fad", "kad", "kld", "tempo", "midi"])
    include_clap: bool = False


class ClassifierTrainRequest(BaseModel):
    real_audio_root: Path
    labels: list[str] | None = None
    output_name: str = "audio_classifier"
    max_files_per_class: int | None = None
    temperature: float = 1.0


class ClassifierPredictRequest(BaseModel):
    model_path: Path | None = None
    audio_paths: list[Path] = Field(default_factory=list)
    audio_root: Path | None = None
