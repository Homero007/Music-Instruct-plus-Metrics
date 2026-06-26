from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.storage.manifest import create_project, load_manifest, list_projects


def test_create_project_manifest(tmp_path):
    config = EngineConfig.for_project_root(tmp_path)

    manifest = create_project(config, "Demo Song")
    loaded = load_manifest(config, manifest.project_id)

    assert loaded.project_id == manifest.project_id
    assert loaded.name == "Demo Song"
    assert (config.projects_dir / manifest.project_id / "source").exists()
    assert len(list_projects(config)) == 1
