import json

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.storage.catalog import (
    list_generations,
    list_renders,
    list_token_manifests,
    list_token_models,
)


def test_resource_catalog_lists_models_generations_and_renders(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    config.ensure_directories()

    manifest_dir = config.tokens_dir / "input" / "tokens-demo"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"token_set_name": "demo", "kind": "input", "total_files": 2}),
        encoding="utf-8",
    )
    empty_manifest_dir = config.tokens_dir / "input" / "tokens-empty"
    empty_manifest_dir.mkdir(parents=True)
    (empty_manifest_dir / "manifest.json").write_text(
        json.dumps({"token_set_name": "empty", "kind": "input", "total_files": 0}),
        encoding="utf-8",
    )

    model_dir = config.data_dir / "models" / "tokens" / "tokenmodel-demo"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_id": "tokenmodel-demo",
                "model_name": "demo_model",
                "order": 2,
                "token_files": 2,
                "states": 12,
            }
        ),
        encoding="utf-8",
    )

    generation_dir = config.data_dir / "generated" / "generated-demo"
    generation_dir.mkdir(parents=True)
    midi_path = generation_dir / "demo.mid"
    midi_path.write_bytes(b"MThd")
    (generation_dir / "tokens.json").write_text(
        json.dumps(
            {
                "generation_id": "generated-demo",
                "duration_seconds_requested": 8,
                "token_count": 20,
                "midi_path": str(midi_path),
            }
        ),
        encoding="utf-8",
    )

    render_dir = config.data_dir / "renders" / "render-demo"
    render_dir.mkdir(parents=True)
    wav_path = render_dir / "render-demo.wav"
    wav_path.write_bytes(b"RIFF")
    (render_dir / "render.json").write_text(
        json.dumps({"engine": "preview", "wav_path": str(wav_path)}),
        encoding="utf-8",
    )

    assert list_token_manifests(config)[0]["path"] == str(manifest_path)
    assert len(list_token_manifests(config)) == 1
    assert list_token_models(config)[0]["path"] == str(model_path)
    assert list_generations(config)[0]["midi_download_url"].startswith("/api/files")
    assert list_renders(config)[0]["wav_download_url"].startswith("/api/files")
