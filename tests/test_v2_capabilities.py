import json
from pathlib import Path

import numpy as np
import pytest
from mido import Message, MidiFile, MidiTrack

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.datasets.augmentation import augment_midi_dataset
from hybrid_music_engine.embeddings.token_vae import (
    encode_token_vae_embedding,
    encode_token_vae_genre_embeddings,
    train_token_vae,
)
from hybrid_music_engine.features.global_features import extract_midi_features
from hybrid_music_engine.fusion.latent_blend import blend_weighted_embedding_files
from hybrid_music_engine.jobs.dispatcher import _ensure_celery_available
from hybrid_music_engine.render.pedalboard_vst import process_audio_with_pedalboard
from hybrid_music_engine.tokens.midi_tokenizer import tokenize_midi_file


def _write_midi(path: Path, note: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("program_change", channel=0, program=0, time=0))
    for pitch in [note, note + 4, note + 7, note + 12]:
        track.append(Message("note_on", channel=0, note=pitch, velocity=80, time=120))
        track.append(Message("note_off", channel=0, note=pitch, velocity=0, time=120))
    midi.save(path)


def _token_manifest(tmp_path: Path, midi_paths: list[Path]) -> Path:
    entries = []
    for index, midi_path in enumerate(midi_paths, start=1):
        payload = tokenize_midi_file(midi_path, genre="demo", clip_id=f"clip_{index}")
        token_path = tmp_path / f"clip_{index}.tokens.json"
        token_path.write_text(json.dumps(payload), encoding="utf-8")
        entries.append({"path": str(token_path)})
    manifest_path = tmp_path / "tokens_manifest.json"
    manifest_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return manifest_path


def test_midi_augmentation_outputs_valid_catalog_and_midi(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    source = tmp_path / "source" / "electronic" / "clip.mid"
    _write_midi(source)

    result = augment_midi_dataset(
        config,
        source_dir=tmp_path / "source",
        output_name="aug_test",
        transpose_steps=[-12, 0, 12],
        velocity_jitter=4,
        timing_jitter_ticks=2,
        quantize_step_ticks=60,
        seed=9,
    )

    assert result["total_augmented"] == 3
    catalog = json.loads(Path(result["catalog_path"]).read_text(encoding="utf-8"))
    assert catalog["counts"]["electronic"] == 3
    for entry in catalog["entries"]:
        midi = MidiFile(entry["source_midi"])
        notes = [msg.note for track in midi.tracks for msg in track if msg.type in {"note_on", "note_off"}]
        assert notes
        assert min(notes) >= 0
        assert max(notes) <= 127


def test_token_vae_trains_and_encodes_embedding(tmp_path):
    pytest.importorskip("torch")
    config = EngineConfig.for_project_root(tmp_path)
    first = tmp_path / "a.mid"
    second = tmp_path / "b.mid"
    _write_midi(first, note=60)
    _write_midi(second, note=65)
    manifest_path = _token_manifest(tmp_path, [first, second])

    metadata = train_token_vae(
        config,
        token_manifest_path=manifest_path,
        latent_dim=4,
        hidden_dim=8,
        epochs=1,
        seed=3,
    )
    embedding = encode_token_vae_embedding(
        config,
        token_source_path=manifest_path,
        model_path=Path(metadata["model_path"]),
        output_name="fixture_embedding",
    )

    assert metadata["sequence_count"] == 2
    assert Path(metadata["model_path"]).exists()
    assert embedding["latent_dim"] == 4
    assert len(embedding["embedding"]) == 4
    assert Path(embedding["path"]).exists()


def test_genre_embeddings_can_be_weighted_for_fusion(tmp_path):
    pytest.importorskip("torch")
    config = EngineConfig.for_project_root(tmp_path)
    entries = []
    for genre, tokens in {
        "electronic": ["genre:electronic", "pitch:60", "dur:short", "vel:high"],
        "reggaeton": ["genre:reggaeton", "pitch:67", "dur:short", "vel:mid"],
        "classical": ["genre:classical", "pitch:72", "dur:long", "vel:soft"],
    }.items():
        token_path = tmp_path / f"{genre}.tokens.json"
        token_path.write_text(json.dumps({"genre": genre, "tokens": tokens}), encoding="utf-8")
        entries.append({"path": str(token_path), "genre": genre})
    manifest_path = tmp_path / "tokens_manifest.json"
    manifest_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    metadata = train_token_vae(
        config,
        token_manifest_path=manifest_path,
        latent_dim=4,
        hidden_dim=8,
        epochs=1,
        seed=3,
    )
    genre_embeddings = encode_token_vae_genre_embeddings(
        config,
        token_manifest_path=manifest_path,
        model_path=Path(metadata["model_path"]),
        output_name="fixture_genres",
    )
    paths_by_genre = {item["genre"]: item["path"] for item in genre_embeddings["embeddings"]}
    blend = blend_weighted_embedding_files(
        config,
        embeddings=[
            {"path": paths_by_genre["electronic"], "weight": 0.5, "label": "electronic"},
            {"path": paths_by_genre["reggaeton"], "weight": 0.3, "label": "reggaeton"},
            {"path": paths_by_genre["classical"], "weight": 0.2, "label": "classical"},
        ],
        output_name="fixture_fusion",
    )

    assert genre_embeddings["genre_count"] == 3
    assert blend["schema_version"] == "weighted-latent-blend-v1"
    assert len(blend["embedding"]) == 4
    assert sum(source["normalized_weight"] for source in blend["sources"]) == pytest.approx(1.0)
    assert Path(blend["path"]).exists()


def test_advanced_midi_features_include_key_rhythm_and_structure(tmp_path):
    midi_path = tmp_path / "feature.mid"
    _write_midi(midi_path)

    features = extract_midi_features(midi_path)

    assert features["pitch_class_profile"]
    assert features["estimated_key"] is not None
    assert "swing_ratio" in features
    assert "syncopation_score" in features
    assert features["chord_hints"]
    assert len(features["groove_vector"]) == 16
    assert features["structure"]


def test_pedalboard_effect_chain_processes_wav(tmp_path):
    pytest.importorskip("pedalboard")
    sf = pytest.importorskip("soundfile")
    sample_rate = 48_000
    seconds = 0.25
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = np.stack([0.1 * np.sin(2 * np.pi * 440 * t), 0.1 * np.sin(2 * np.pi * 440 * t)], axis=1)
    input_wav = tmp_path / "input.wav"
    output_wav = tmp_path / "master.wav"
    sf.write(input_wav, audio, sample_rate)

    result = process_audio_with_pedalboard(input_wav, output_wav, preset="master")

    assert result["engine"] == "pedalboard-effects"
    assert Path(result["wav_path"]).exists()


def test_require_celery_rejects_unavailable_broker(monkeypatch):
    monkeypatch.setenv("HYBRID_ENGINE_REQUIRE_CELERY", "1")
    monkeypatch.setenv("HYBRID_ENGINE_BROKER_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("HYBRID_ENGINE_RESULT_BACKEND", "redis://127.0.0.1:1/0")

    with pytest.raises(RuntimeError, match="Celery/Redis es obligatorio"):
        _ensure_celery_available()
