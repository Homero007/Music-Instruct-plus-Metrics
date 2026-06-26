import json
from pathlib import Path

from mido import Message, MidiFile, MidiTrack

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.fusion.latent_blend import blend_embedding_files
from hybrid_music_engine.generation.ranked import generate_ranked_candidates
from hybrid_music_engine.render.pedalboard_engine import (
    mix_layer_renders,
    render_midi_audio,
    render_midi_fluidsynth_wav,
    render_midi_preview_wav,
)
from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
from hybrid_music_engine.tokens.generative_model import (
    generate_tokens_from_model,
    train_token_markov_model,
    tokens_to_midi,
)
from hybrid_music_engine.tokens.midi_tokenizer import tokenize_midi_file
from hybrid_music_engine.tokens.transformer_model import train_token_transformer_model


def _write_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("program_change", channel=0, program=0, time=0))
    for note in [60, 64, 67, 72]:
        track.append(Message("note_on", channel=0, note=note, velocity=80, time=120))
        track.append(Message("note_off", channel=0, note=note, velocity=0, time=120))
    midi.save(path)


def test_train_token_model_generate_midi_and_render(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)
    token_payload = tokenize_midi_file(midi_path, genre="demo", clip_id="clip_1")
    token_path = tmp_path / "clip_1.tokens.json"
    token_path.write_text(json.dumps(token_payload), encoding="utf-8")
    manifest_path = tmp_path / "tokens_manifest.json"
    manifest_path.write_text(
        json.dumps({"entries": [{"path": str(token_path)}]}),
        encoding="utf-8",
    )

    model = train_token_markov_model(config, token_manifest_path=manifest_path, order=1)
    generated = generate_tokens_from_model(
        config,
        model_path=Path(model["path"]),
        duration_seconds=4,
        seed=7,
    )
    render = render_midi_preview_wav(
        Path(generated["midi_path"]),
        tmp_path / "preview.wav",
    )

    assert Path(model["path"]).exists()
    assert Path(generated["midi_path"]).exists()
    assert Path(render["wav_path"]).exists()
    assert render["note_count"] >= 1


def test_train_transformer_token_model_and_generate_midi(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)
    token_payload = tokenize_midi_file(midi_path, genre="demo", clip_id="clip_1")
    token_path = tmp_path / "clip_1.tokens.json"
    token_path.write_text(json.dumps(token_payload), encoding="utf-8")
    manifest_path = tmp_path / "tokens_manifest.json"
    manifest_path.write_text(
        json.dumps({"entries": [{"path": str(token_path)}]}),
        encoding="utf-8",
    )

    model = train_token_transformer_model(
        config,
        token_manifest_path=manifest_path,
        model_name="tiny_transformer",
        sequence_length=16,
        epochs=1,
        batch_size=2,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
    )
    embedding_path = tmp_path / "embedding.json"
    embedding_path.write_text(json.dumps({"embedding": [0.2, -0.4, 0.8]}), encoding="utf-8")
    generated = generate_tokens_from_model(
        config,
        model_path=Path(model["path"]),
        duration_seconds=2,
        seed=11,
        max_tokens=24,
        condition_genre="demo",
        feature_tokens=["density:low"],
        embedding_path=embedding_path,
        export_layers=True,
    )

    assert model["model_type"] == "transformer"
    assert Path(model["checkpoint_path"]).exists()
    assert generated["generator"] == "token-transformer"
    assert generated["condition_genre"] == "demo"
    assert generated["embedding_path"] == str(embedding_path.resolve())
    assert generated["metrics"]["valid_midi"] is True
    assert Path(generated["midi_path"]).exists()
    assert Path(generated["layer_midis"]["melody"]).exists()


def test_generated_remi_midi_respects_requested_duration(tmp_path):
    midi_path = tmp_path / "duration.mid"
    tokens = [
        "ticks_per_beat:480",
        "tempo:500000",
        "bar:0",
        "position:0",
        "program:0:0",
        "layer:melody",
        "note:0:60:10:240",
        "position:4",
        "note:0:64:10:240",
    ]

    tokens_to_midi(tokens, midi_path, duration_seconds=30)
    metrics = analyze_midi_quality(midi_path)

    assert metrics["valid_midi"] is True
    assert metrics["duration_seconds"] == 30
    assert metrics["note_count"] > 2


def test_midi_quality_metrics(tmp_path):
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)

    metrics = analyze_midi_quality(midi_path)

    assert metrics["valid_midi"] is True
    assert metrics["note_count"] == 4
    assert metrics["quality_score"] >= 0


def test_generate_ranked_candidates(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)
    token_payload = tokenize_midi_file(midi_path, genre="demo", clip_id="clip_1")
    token_path = tmp_path / "clip_1.tokens.json"
    token_path.write_text(json.dumps(token_payload), encoding="utf-8")
    manifest_path = tmp_path / "tokens_manifest.json"
    manifest_path.write_text(
        json.dumps({"entries": [{"path": str(token_path)}]}),
        encoding="utf-8",
    )
    model = train_token_markov_model(config, token_manifest_path=manifest_path, order=1)

    ranking = generate_ranked_candidates(
        config,
        model_path=Path(model["path"]),
        duration_seconds=2,
        output_name="ranked_test",
        candidates=3,
        seed=3,
        export_layers=True,
    )

    scores = [candidate["score"] for candidate in ranking["candidates"]]
    assert ranking["best_candidate_id"] == ranking["candidates"][0]["candidate_id"]
    assert scores == sorted(scores, reverse=True)
    assert Path(ranking["path"]).exists()


def test_blend_embedding_files(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"embedding": [1.0, 0.0]}), encoding="utf-8")
    b.write_text(json.dumps({"embedding": [0.0, 1.0]}), encoding="utf-8")

    blended = blend_embedding_files(
        config,
        embedding_a_path=a,
        embedding_b_path=b,
        alpha=0.25,
    )

    assert blended["embedding"] == [0.25, 0.75]
    assert Path(blended["path"]).exists()


def test_render_auto_falls_back_to_preview_without_soundfont(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)
    monkeypatch.setattr(
        "hybrid_music_engine.render.pedalboard_engine.shutil.which",
        lambda _binary: None,
    )

    render = render_midi_audio(
        midi_path,
        tmp_path / "renders",
        config=config,
        engine="auto",
    )

    assert render["requested_engine"] == "auto"
    assert render["engine"] == "internal-sine-preview"
    assert Path(render["wav_path"]).exists()


def test_mix_layer_renders(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)
    renders = {
        "melody": render_midi_preview_wav(midi_path, tmp_path / "melody.wav"),
        "bass": render_midi_preview_wav(midi_path, tmp_path / "bass.wav"),
    }

    mix = mix_layer_renders(renders, tmp_path / "mix", config=config)

    assert mix["engine"] == "layer-mix-master"
    assert Path(mix["wav_path"]).exists()


def test_fluidsynth_requires_binary_or_soundfont(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "input.mid"
    _write_midi(midi_path)
    monkeypatch.setattr(
        "hybrid_music_engine.render.pedalboard_engine.shutil.which",
        lambda _binary: None,
    )

    try:
        render_midi_fluidsynth_wav(
            midi_path,
            tmp_path / "out.wav",
            config=config,
        )
    except RuntimeError as exc:
        assert "FluidSynth" in str(exc)
    else:
        raise AssertionError("FluidSynth render debió fallar sin binario.")
