from pathlib import Path

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.jobs import workflows
from hybrid_music_engine.storage.job_store import create_job, load_job
from hybrid_music_engine.storage.manifest import create_project, load_manifest, project_path, save_manifest


def test_transcribe_drums_workflow_updates_manifest(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    manifest = create_project(config, "Drums Test")
    stems_dir = project_path(config, manifest.project_id) / "stems"
    drums = stems_dir / "drums.wav"
    drums.parent.mkdir(parents=True, exist_ok=True)
    drums.write_bytes(b"drums")
    manifest.stems = {"files": {"drums": str(drums)}}
    save_manifest(config, manifest)
    job = create_job(config, kind="transcribe-drums", project_id=manifest.project_id)

    monkeypatch.setattr(workflows.EngineConfig, "from_env", classmethod(lambda cls: config))

    def fake_transcribe(source_audio: Path, output_midi: Path, **kwargs):
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        output_midi.write_bytes(b"midi")
        return {
            "source": str(source_audio),
            "midi": str(output_midi),
            "engine": "librosa-onsets",
            "note_count": 3,
            **kwargs,
        }

    monkeypatch.setattr(workflows, "transcribe_drums_onsets", fake_transcribe)

    result = workflows.run_transcribe_drums_job(
        job.job_id,
        manifest.project_id,
        bpm=95,
        onset_delta=0.08,
    )

    updated = load_manifest(config, manifest.project_id)
    updated_job = load_job(config, job.job_id)
    assert updated.status == "drums_midi_transcribed"
    assert updated.midis["drums"]["midi"].endswith("drums.mid")
    assert result["midi"]["engine"] == "librosa-onsets"
    assert result["midi"]["bpm"] == 95
    assert updated_job.status == "completed"
