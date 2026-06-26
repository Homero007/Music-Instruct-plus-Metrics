from pathlib import Path

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.jobs import workflows
from hybrid_music_engine.storage.job_store import create_job, load_job
from hybrid_music_engine.storage.manifest import create_project, load_manifest, project_path, save_manifest


def test_transcribe_melodic_workflow_updates_manifest(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    manifest = create_project(config, "Melody Test")
    stems_dir = project_path(config, manifest.project_id) / "stems"
    bass = stems_dir / "bass.wav"
    bass.parent.mkdir(parents=True, exist_ok=True)
    bass.write_bytes(b"bass")
    manifest.stems = {"files": {"bass": str(bass)}}
    save_manifest(config, manifest)
    job = create_job(config, kind="transcribe-melodic", project_id=manifest.project_id)

    monkeypatch.setattr(workflows.EngineConfig, "from_env", classmethod(lambda cls: config))

    def fake_transcribe(source_audio: Path, output_midi: Path, **kwargs):
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        output_midi.write_bytes(b"midi")
        return {
            "source": str(source_audio),
            "midi": str(output_midi),
            "engine": "basic-pitch",
            **kwargs,
        }

    monkeypatch.setattr(workflows, "transcribe_melodic_basic_pitch", fake_transcribe)

    result = workflows.run_transcribe_melodic_job(
        job.job_id,
        manifest.project_id,
        stems=["bass"],
        minimum_note_length=0.05,
    )

    updated = load_manifest(config, manifest.project_id)
    updated_job = load_job(config, job.job_id)
    assert updated.status == "melodic_midis_transcribed"
    assert updated.midis["melodic"]["bass"]["midi"].endswith("bass.mid")
    assert result["midis"]["bass"]["engine"] == "basic-pitch"
    assert updated_job.status == "completed"
