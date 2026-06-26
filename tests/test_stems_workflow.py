from pathlib import Path

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.audio import stems
from hybrid_music_engine.storage.job_store import create_job, load_job
from hybrid_music_engine.storage.manifest import create_project, load_manifest, project_path, save_manifest
from hybrid_music_engine.jobs import workflows


def test_separate_stems_workflow_updates_manifest(tmp_path, monkeypatch):
    config = EngineConfig.for_project_root(tmp_path)
    manifest = create_project(config, "Stem Test")
    audio_path = project_path(config, manifest.project_id) / "source" / "normalized.wav"
    audio_path.write_bytes(b"wav")
    manifest.source = {"normalized": str(audio_path)}
    save_manifest(config, manifest)
    job = create_job(config, kind="separate-stems", project_id=manifest.project_id)

    monkeypatch.setattr(workflows.EngineConfig, "from_env", classmethod(lambda cls: config))

    def fake_separate_stems(source_audio: Path, stems_dir: Path, **kwargs):
        drums = stems_dir / "drums.wav"
        drums.parent.mkdir(parents=True, exist_ok=True)
        drums.write_bytes(b"drums")
        return {
            "source": str(source_audio),
            "model": kwargs["model_name"],
            "device": kwargs["device"],
            "sample_rate": 44100,
            "files": {"drums": str(drums)},
            "missing": ["bass", "other", "vocals"],
        }

    monkeypatch.setattr(workflows, "separate_stems", fake_separate_stems)

    result = workflows.run_separate_stems_job(
        job.job_id,
        manifest.project_id,
        model_name="htdemucs",
        device="cpu",
    )

    updated = load_manifest(config, manifest.project_id)
    updated_job = load_job(config, job.job_id)
    assert updated.status == "stems_separated"
    assert updated.stems["files"]["drums"].endswith("drums.wav")
    assert result["stems"]["model"] == "htdemucs"
    assert updated_job.status == "completed"


def test_separate_stems_uses_demucs_cli_when_api_is_unavailable(tmp_path, monkeypatch):
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"wav")
    output_dir = tmp_path / "stems"

    def fake_run(command, **_kwargs):
        output_root = Path(command[command.index("-o") + 1])
        model_name = command[command.index("-n") + 1]
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True)
        for stem_name in ["drums", "bass", "vocals", "other"]:
            (model_dir / f"{stem_name}.wav").write_bytes(stem_name.encode())

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(stems.subprocess, "run", fake_run)

    result = stems.separate_stems(audio_path, output_dir, model_name="htdemucs", device="cpu")

    assert result["engine"] == "demucs-cli"
    assert result["missing"] == []
    assert Path(result["files"]["drums"]).exists()
