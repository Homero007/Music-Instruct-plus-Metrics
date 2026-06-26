import json

import pytest

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.embeddings.vae_model import (
    encode_project_embedding,
    flatten_numeric_features,
    train_feature_vae,
    vectorize_features,
)
from hybrid_music_engine.storage.manifest import create_project, load_manifest, project_path, save_manifest


def test_feature_vectorization_is_stable():
    payload = {
        "audio": {"tempo": 120.0, "valid": True},
        "midi": {"notes": [60, 64, 67], "name": "ignored"},
    }

    flat = flatten_numeric_features(payload)
    vector, names, diagnostics = vectorize_features(payload, ["audio.tempo", "midi.notes[1]"])

    assert flat["audio.valid"] == 1.0
    assert names == ["audio.tempo", "midi.notes[1]"]
    assert vector.tolist() == [120.0, 64.0]
    assert diagnostics["missing_features"] == []


def test_train_and_encode_feature_vae(tmp_path):
    pytest.importorskip("torch")
    config = EngineConfig.for_project_root(tmp_path)
    manifest = create_project(config, "Embedding Test")
    features_path = project_path(config, manifest.project_id) / "features" / "features.json"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(
        json.dumps(
            {
                "audio": {"tempo_bpm": 100.0, "duration_seconds": 30.0},
                "midi": {"drums": {"note_count": 24, "pitch_diversity": 3}},
                "summary": {"total_midi_notes": 24},
            }
        ),
        encoding="utf-8",
    )
    manifest.features = {"path": str(features_path)}
    save_manifest(config, manifest)

    metadata = train_feature_vae(
        config,
        latent_dim=4,
        hidden_dim=8,
        epochs=2,
        learning_rate=1e-3,
    )
    encoded = encode_project_embedding(load_manifest(config, manifest.project_id), config)

    assert metadata["latent_dim"] == 4
    assert metadata["project_count"] == 1
    assert len(encoded["embedding"]) == 4
    assert encoded["path"].endswith("embedding.json")
