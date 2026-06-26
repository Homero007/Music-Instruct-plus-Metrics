from pathlib import Path
import json

import numpy as np
import soundfile as sf

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.datasets import clip_processor
from hybrid_music_engine.datasets import jamendo


def test_download_jamendo_catalog_with_configurable_genres(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)

    def fake_query(*, tags, limit, offset, **_kwargs):
        if offset > 0:
            return {"results": []}
        return {
            "results": [
                {
                    "id": f"{tags[0]}-1",
                    "name": "Track",
                    "artist_name": "Artist",
                    "duration": "180",
                    "audiodownload": f"https://example.test/{tags[0]}.mp3",
                    "license_ccurl": "https://creativecommons.org/licenses/by/4.0/",
                }
            ][:limit]
        }

    def fake_download(_url: str, target_path: Path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"audio")

    monkeypatch.setattr(jamendo, "query_jamendo_tracks", fake_query)
    monkeypatch.setattr(jamendo, "download_file", fake_download)

    catalog = jamendo.download_jamendo_catalog(
        config,
        genre_tags={
            "classical": ["classical"],
            "electronic": ["electronic"],
            "reggaeton": ["reggaeton"],
            "salsa": ["salsa"],
        },
        max_tracks_per_genre=None,
        download_audio=True,
        source="api",
    )

    assert catalog["genre_count"] == 4
    assert catalog["total_tracks"] == 4
    assert catalog["counts"] == {
        "classical": 1,
        "electronic": 1,
        "reggaeton": 1,
        "salsa": 1,
    }
    assert all(entry["audio_path"] for entry in catalog["entries"])
    assert Path(catalog["path"]).exists()


def test_download_jamendo_catalog_can_be_metadata_only(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)

    def fake_query(*, offset, **_kwargs):
        if offset > 0:
            return {"results": []}
        return {"results": [{"id": "1", "name": "Track", "duration": "60"}]}

    monkeypatch.setattr(jamendo, "query_jamendo_tracks", fake_query)

    catalog = jamendo.download_jamendo_catalog(
        config,
        genre_tags={"ambient": ["ambient"]},
        download_audio=False,
        source="api",
    )

    assert catalog["total_tracks"] == 1
    assert catalog["entries"][0]["audio_path"] is None
    assert catalog["entries"][0]["download_error"] is None


def test_prepare_jamendo_clips_from_downloaded_audio(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    sample_rate = 16000
    audio_path = tmp_path / "audio" / "electronic" / "track-1.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(sample_rate * 3, dtype=np.float32)
    audio[100:300] = 0.5
    sf.write(audio_path, audio, sample_rate)

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "jamendo_test",
                "entries": [
                    {
                        "track_id": "track-1",
                        "genre": "electronic",
                        "audio_path": str(audio_path),
                        "name": "Track",
                        "artist_name": "Artist",
                        "license_ccurl": "https://creativecommons.org/licenses/by/4.0/",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    clips = jamendo.prepare_jamendo_clips(
        config,
        catalog_path=catalog_path,
        clip_duration_seconds=1.0,
        hop_duration_seconds=1.0,
        max_clips_per_track=2,
        min_clip_seconds=0.5,
        sample_rate=sample_rate,
    )

    assert clips["total_clips"] == 2
    assert clips["counts"] == {"electronic": 2}
    assert Path(clips["entries"][0]["clip_path"]).exists()
    assert Path(clips["path"]).exists()


def test_prepare_jamendo_clips_repairs_duplicate_project_root_audio_path(tmp_path):
    config = EngineConfig.for_project_root(tmp_path / "hybrid_engine")
    config.ensure_directories()
    sample_rate = 16000
    audio_path = config.project_root / "data" / "datasets" / "jamendo" / "audio" / "electronic" / "track-1.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, np.ones(sample_rate, dtype=np.float32) * 0.1, sample_rate)
    duplicated_path = str(config.project_root / config.project_root.name / "data" / "datasets" / "jamendo" / "audio" / "electronic" / "track-1.wav")

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "duplicate-root",
                "entries": [
                    {
                        "track_id": "track-1",
                        "genre": "electronic",
                        "audio_path": duplicated_path,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    clips = jamendo.prepare_jamendo_clips(
        config,
        catalog_path=catalog_path,
        clip_duration_seconds=0.5,
        max_clips_per_track=1,
        min_clip_seconds=0.25,
        sample_rate=sample_rate,
    )

    assert clips["total_clips"] == 1
    assert Path(clips["entries"][0]["clip_path"]).exists()


def test_prepare_jamendo_clips_fails_when_all_audio_paths_are_missing(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "missing-audio",
                "entries": [
                    {
                        "track_id": "track-1",
                        "genre": "electronic",
                        "audio_path": str(tmp_path / "missing.mp3"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        jamendo.prepare_jamendo_clips(
            config,
            catalog_path=catalog_path,
            clip_duration_seconds=1,
            min_clip_seconds=0.5,
        )
    except RuntimeError as exc:
        assert "No se pudo crear ningún clip" in str(exc)
        assert "audio_file_not_found" in str(exc)
    else:
        raise AssertionError("Preparar clips debía fallar con catálogo vacío.")


def test_select_jamendo_catalog_entries_by_genre_and_count(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    e1 = audio_dir / "e1.mp3"
    e2 = audio_dir / "e2.mp3"
    j1 = audio_dir / "j1.mp3"
    for path in [e1, e2, j1]:
        path.write_bytes(b"audio")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "source",
                "counts": {"electronic": 2, "jazz": 1},
                "entries": [
                    {"track_id": "e1", "genre": "electronic", "audio_path": str(e1)},
                    {"track_id": "e2", "genre": "electronic", "audio_path": str(e2)},
                    {"track_id": "j1", "genre": "jazz", "audio_path": str(j1)},
                ],
            }
        ),
        encoding="utf-8",
    )

    selected = jamendo.select_jamendo_catalog_entries(
        config,
        catalog_path=catalog_path,
        genres=["electronic"],
        max_tracks_per_genre=1,
        output_name="demo",
    )

    assert selected["counts"] == {"electronic": 1}
    assert selected["total_tracks"] == 1
    assert selected["entries"][0]["track_id"] == "e1"
    assert Path(selected["entries"][0]["audio_path"]).is_absolute()
    assert Path(selected["path"]).exists()


def test_select_jamendo_catalog_entries_fails_without_local_audio(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "source",
                "counts": {"electronic": 1},
                "entries": [
                    {
                        "track_id": "e1",
                        "genre": "electronic",
                        "audio_path": str(tmp_path / "missing.mp3"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        jamendo.select_jamendo_catalog_entries(
            config,
            catalog_path=catalog_path,
            genres=["electronic"],
            max_tracks_per_genre=1,
        )
    except RuntimeError as exc:
        assert "No hay audios locales utilizables" in str(exc)
    else:
        raise AssertionError("La selección debía fallar sin audio local.")


def test_process_jamendo_clips_continues_when_stems_are_unavailable(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    sample_rate = 16000
    clip_path = tmp_path / "clips" / "electronic" / "clip.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.sin(np.linspace(0, 8 * np.pi, sample_rate, dtype=np.float32)) * 0.2
    sf.write(clip_path, audio, sample_rate)

    clips_catalog_path = tmp_path / "clips_catalog.json"
    clips_catalog_path.write_text(
        json.dumps(
            {
                "source_catalog_id": "jamendo_test",
                "entries": [
                    {
                        "clip_id": "clip-1",
                        "genre": "electronic",
                        "clip_path": str(clip_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_separate_stems(*_args, **_kwargs):
        raise RuntimeError("Demucs no está instalado.")

    def fake_transcribe(_audio_path: Path, output_midi: Path):
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

        midi = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        midi.tracks.append(track)
        track.append(MetaMessage("set_tempo", tempo=bpm2tempo(120), time=0))
        track.append(Message("program_change", program=0, channel=0, time=0))
        track.append(Message("note_on", note=60, velocity=80, channel=0, time=0))
        track.append(Message("note_off", note=60, velocity=0, channel=0, time=480))
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        midi.save(output_midi)
        return {"midi": str(output_midi), "engine": "fake"}

    monkeypatch.setattr(clip_processor, "separate_stems", fake_separate_stems)
    monkeypatch.setattr(clip_processor, "transcribe_melodic_basic_pitch", fake_transcribe)

    batch = clip_processor.process_jamendo_clips(
        config,
        clips_catalog_path=clips_catalog_path,
        run_stems=True,
        run_melodic=True,
        run_drums=False,
        run_features=True,
        run_tokens=True,
    )

    assert batch["total_completed"] == 1
    assert "No se pudieron separar stems" in batch["entries"][0]["warnings"][0]
    token_manifest = json.loads(Path(batch["token_manifest_path"]).read_text(encoding="utf-8"))
    assert token_manifest["total_files"] == 1


def test_token_vae_demucs_mode_requires_stems(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    clip_path = _write_clip(tmp_path)
    clips_catalog_path = _write_clips_catalog(tmp_path, clip_path)

    def fake_separate_stems(*_args, **_kwargs):
        raise RuntimeError("Demucs no está instalado.")

    monkeypatch.setattr(clip_processor, "separate_stems", fake_separate_stems)

    try:
        clip_processor.process_jamendo_clips(
            config,
            clips_catalog_path=clips_catalog_path,
            processing_mode="token_vae_demucs",
            continue_on_error=True,
        )
    except RuntimeError as exc:
        assert "Token-VAE Demucs no generó tokens" in str(exc)
        assert "Demucs no está instalado" in str(exc)
    else:
        raise AssertionError("El modo serio debía fallar claramente sin Demucs.")


def test_token_vae_demucs_mode_writes_serious_manifest(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    clip_path = _write_clip(tmp_path)
    clips_catalog_path = _write_clips_catalog(tmp_path, clip_path)

    def fake_separate_stems(_audio_path: Path, output_dir: Path, **_kwargs):
        files = {}
        for name in ["drums", "bass", "vocals", "other"]:
            stem = output_dir / f"{name}.wav"
            stem.parent.mkdir(parents=True, exist_ok=True)
            sf.write(stem, np.ones(1600, dtype=np.float32) * 0.05, 16000)
            files[name] = str(stem)
        return {"files": files, "engine": "fake-demucs"}

    def fake_transcribe(_audio_path: Path, output_midi: Path, **_kwargs):
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

        midi = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        midi.tracks.append(track)
        track.append(MetaMessage("set_tempo", tempo=bpm2tempo(120), time=0))
        track.append(Message("program_change", program=0, channel=0, time=0))
        track.append(Message("note_on", note=52, velocity=80, channel=0, time=0))
        track.append(Message("note_off", note=52, velocity=0, channel=0, time=240))
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        midi.save(output_midi)
        return {"midi": str(output_midi), "engine": "fake"}

    monkeypatch.setattr(clip_processor, "separate_stems", fake_separate_stems)
    monkeypatch.setattr(clip_processor, "transcribe_melodic_basic_pitch", fake_transcribe)
    monkeypatch.setattr(clip_processor, "transcribe_drums_onsets", fake_transcribe)

    batch = clip_processor.process_jamendo_clips(
        config,
        clips_catalog_path=clips_catalog_path,
        processing_mode="token_vae_demucs",
    )

    assert batch["total_completed"] == 1
    token_manifest = json.loads(Path(batch["token_manifest_path"]).read_text(encoding="utf-8"))
    assert token_manifest["processing_mode"] == "token_vae_demucs"
    assert token_manifest["intended_model"] == "token_vae"
    assert {entry["layer"] for entry in token_manifest["entries"]} >= {"bass", "drums", "harmony", "melody"}


def _write_clip(tmp_path: Path) -> Path:
    sample_rate = 16000
    clip_path = tmp_path / "clips" / "electronic" / "clip.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.sin(np.linspace(0, 8 * np.pi, sample_rate, dtype=np.float32)) * 0.2
    sf.write(clip_path, audio, sample_rate)
    return clip_path


def _write_clips_catalog(tmp_path: Path, clip_path: Path) -> Path:
    clips_catalog_path = tmp_path / "clips_catalog.json"
    clips_catalog_path.write_text(
        json.dumps(
            {
                "source_catalog_id": "jamendo_test",
                "entries": [
                    {
                        "clip_id": "clip-1",
                        "genre": "electronic",
                        "clip_path": str(clip_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return clips_catalog_path
