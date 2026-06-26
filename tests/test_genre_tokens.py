import json
from pathlib import Path
from zipfile import ZipFile

from mido import Message, MidiFile, MidiTrack

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.datasets.cleaning import clean_midi_dataset
from hybrid_music_engine.datasets.genre_catalog import build_genre_catalog
from hybrid_music_engine.tokens.midi_tokenizer import (
    create_generation_plan,
    export_token_manifest_to_zip,
    export_output_tokens_to_zip,
    tokenize_catalog_to_zip,
    tokenize_midi_file,
)


def _write_midi(path: Path, note: int = 60, ticks: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("program_change", channel=0, program=0, time=0))
    track.append(Message("note_on", channel=0, note=note, velocity=80, time=0))
    track.append(Message("note_off", channel=0, note=note, velocity=0, time=ticks))
    midi.save(path)


def test_build_genre_catalog_and_token_zip(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    source = tmp_path / "source_midis"
    for genre_index, genre in enumerate(["rock", "jazz", "pop"]):
        for clip_index in range(2):
            _write_midi(source / genre / f"clip_{clip_index}.mid", note=60 + genre_index)

    catalog = build_genre_catalog(
        config,
        source_dir=source,
        genres=["rock", "jazz", "pop"],
        clips_per_genre=2,
        max_duration_seconds=10,
    )
    token_set = tokenize_catalog_to_zip(config, catalog_path=Path(catalog["path"]))

    assert catalog["counts"] == {"rock": 2, "jazz": 2, "pop": 2}
    assert token_set["counts"] == {"jazz": 2, "pop": 2, "rock": 2}
    assert Path(token_set["zip_path"]).exists()
    with ZipFile(token_set["zip_path"]) as archive:
        names = archive.namelist()
    assert "manifest.json" in names
    assert any(name.startswith("genres/rock/") for name in names)


def test_clean_midi_dataset_filters_and_copies(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    source = tmp_path / "dirty"
    _write_midi(source / "ok.mid", note=62, ticks=960)
    (source / "broken.mid").write_text("not midi", encoding="utf-8")

    clean = clean_midi_dataset(
        config,
        source_dir=source,
        output_name="unit_clean",
        min_duration_seconds=0.1,
        min_notes=1,
        min_quality_score=0,
    )

    assert clean["accepted_count"] == 1
    assert clean["rejected_count"] == 1
    assert Path(clean["accepted"][0]["clean_midi"]).exists()


def test_build_catalog_accepts_more_than_three_genres(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    source = tmp_path / "source_midis"
    genres = ["rock", "jazz", "pop", "salsa"]
    for genre_index, genre in enumerate(genres):
        _write_midi(source / genre / "clip_0.mid", note=60 + genre_index)

    catalog = build_genre_catalog(
        config,
        source_dir=source,
        genres=genres,
        clips_per_genre=1,
        max_duration_seconds=10,
    )

    assert catalog["genre_count"] == 4
    assert catalog["counts"] == {"rock": 1, "jazz": 1, "pop": 1, "salsa": 1}


def test_tokenize_midi_and_export_output_tokens(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    midi_path = tmp_path / "mixed" / "candidate.mid"
    _write_midi(midi_path)

    token_payload = tokenize_midi_file(midi_path, genre="mixed", clip_id="candidate")
    token_path = midi_path.with_suffix(".tokens.json")
    token_path.write_text(json.dumps(token_payload), encoding="utf-8")
    export = export_output_tokens_to_zip(
        config,
        source_dir=midi_path.parent,
        export_name="mixed_test",
        duration_seconds=24,
    )

    assert token_payload["token_count"] > 0
    assert token_payload["tokenizer"]["structure_tokens"] is True
    assert any(token.startswith("bar:") for token in token_payload["tokens"])
    assert any(token.startswith("position:") for token in token_payload["tokens"])
    assert export["requested_duration_seconds"] == 24
    assert export["counts"] == {"mixed": 1}
    assert Path(export["zip_path"]).exists()
    with ZipFile(export["zip_path"]) as archive:
        names = archive.namelist()
    assert "manifest.json" in names
    assert any(name.startswith("genres/mixed/") for name in names)


def test_export_input_tokens_from_manifest_keeps_genre_folders(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    entries = []
    for genre in ["electronic", "reggaeton"]:
        token_path = tmp_path / "tokens" / genre / "clip.tokens.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(
            json.dumps({"clip_id": f"{genre}_clip", "genre": genre, "token_count": 4, "tokens": ["a", "b", "c", "d"]}),
            encoding="utf-8",
        )
        entries.append({"path": str(token_path), "genre": genre, "token_count": 4})
    manifest_path = tmp_path / "tokens_manifest.json"
    manifest_path.write_text(json.dumps({"entries": entries, "total_files": 2}), encoding="utf-8")

    export = export_token_manifest_to_zip(
        config,
        token_manifest_path=manifest_path,
        export_name="input_manifest_test",
    )

    assert export["counts"] == {"electronic": 1, "reggaeton": 1}
    with ZipFile(export["zip_path"]) as archive:
        names = archive.namelist()
    assert any(name.startswith("genres/electronic/") for name in names)
    assert any(name.startswith("genres/reggaeton/") for name in names)


def test_generation_plan_records_duration(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    plan = create_generation_plan(
        config,
        project_id="project_demo",
        duration_seconds=42,
        output_name="demo_output",
    )

    assert plan["duration_seconds"] == 42
    assert Path(plan["path"]).exists()
