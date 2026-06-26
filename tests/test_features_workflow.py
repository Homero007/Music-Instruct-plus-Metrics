from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.features.global_features import extract_midi_features
from hybrid_music_engine.jobs import workflows
from hybrid_music_engine.storage.job_store import create_job, load_job
from hybrid_music_engine.storage.manifest import create_project, load_manifest, project_path, save_manifest


def test_extract_features_workflow_updates_manifest(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    manifest = create_project(config, "Features Test")
    project_dir = project_path(config, manifest.project_id)
    audio = project_dir / "source" / "normalized.wav"
    audio.write_bytes(b"audio")
    midi = project_dir / "midis" / "drums.mid"
    midi.write_bytes(b"midi")
    manifest.source = {"normalized": str(audio)}
    manifest.midis = {"drums": {"midi": str(midi)}}
    save_manifest(config, manifest)
    job = create_job(config, kind="extract-features", project_id=manifest.project_id)

    monkeypatch.setattr(workflows.EngineConfig, "from_env", classmethod(lambda cls: config))

    def fake_extract_project_features(manifest_arg, config_arg, **kwargs):
        output = project_path(config_arg, manifest_arg.project_id) / "features" / "features.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return {
            "project_id": manifest_arg.project_id,
            "path": str(output),
            "audio": {"normalized": {"duration_seconds": 10}},
            "midi": {"drums": {"note_count": 4}},
            "summary": {"total_midi_notes": 4},
            **kwargs,
        }

    monkeypatch.setattr(workflows, "extract_project_features", fake_extract_project_features)

    result = workflows.run_extract_features_job(job.job_id, manifest.project_id)

    updated = load_manifest(config, manifest.project_id)
    updated_job = load_job(config, job.job_id)
    assert updated.status == "features_extracted"
    assert updated.features["summary"]["total_midi_notes"] == 4
    assert result["features"]["path"].endswith("features.json")
    assert updated_job.status == "completed"


def test_extract_midi_features_reads_basic_notes(tmp_path):
    from mido import Message, MidiFile, MidiTrack

    midi_path = tmp_path / "layer.mid"
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("program_change", channel=0, program=1, time=0))
    track.append(Message("note_on", channel=0, note=60, velocity=80, time=0))
    track.append(Message("note_off", channel=0, note=60, velocity=0, time=480))
    midi.save(midi_path)

    features = extract_midi_features(midi_path)

    assert features["note_count"] == 1
    assert features["pitch_min"] == 60
    assert features["pitch_max"] == 60
    assert features["programs"] == [1]
