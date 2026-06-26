from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class KeepFile:
    genre: str
    source: Path
    target: Path
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a compact delivery copy in-place: keep a fixed number of Jamendo "
            "songs per genre and remove generated training/test artifacts."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete/copy files. Without this flag the script only prints a dry run.",
    )
    parser.add_argument(
        "--keep-per-genre",
        type=int,
        default=150,
        help="Number of downloaded songs to keep for each genre.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the project root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--source-audio-dir",
        type=Path,
        default=None,
        help="Source Jamendo audio directory. Defaults to the known downloaded catalog.",
    )
    parser.add_argument(
        "--output-name",
        default="delivery_jamendo_150",
        help="Name of the compact Jamendo dataset folder to create.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def load_existing_metadata(project_root: Path) -> dict[str, dict]:
    catalog_path = (
        project_root
        / "data"
        / "datasets"
        / "jamendo"
        / "existing_downloads_catalog"
        / "catalog.json"
    )
    if not catalog_path.exists():
        return {}
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("entries", [])
    metadata: dict[str, dict] = {}
    for entry in entries:
        audio_path = entry.get("audio_path") or entry.get("source_file") or entry.get("path")
        if not audio_path:
            continue
        key = f"{entry.get('genre', 'unknown')}::{Path(str(audio_path)).name}"
        metadata[key] = entry
    return metadata


def select_files(source_audio_dir: Path, output_audio_dir: Path, keep_per_genre: int) -> list[KeepFile]:
    keep_files: list[KeepFile] = []
    for genre_dir in sorted(source_audio_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        files = sorted(
            [
                path
                for path in genre_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
            ],
            key=lambda item: item.name,
        )
        selected = files[:keep_per_genre]
        for source in selected:
            target = output_audio_dir / genre_dir.name / source.name
            keep_files.append(
                KeepFile(
                    genre=genre_dir.name,
                    source=source,
                    target=target,
                    size_bytes=source.stat().st_size,
                )
            )
    return keep_files


def build_catalog(project_root: Path, output_dir: Path, keep_files: list[KeepFile]) -> dict:
    metadata = load_existing_metadata(project_root)
    entries: list[dict] = []
    counts: dict[str, int] = {}
    for index, item in enumerate(keep_files, start=1):
        counts[item.genre] = counts.get(item.genre, 0) + 1
        key = f"{item.genre}::{item.source.name}"
        source_meta = metadata.get(key, {})
        audio_rel = rel(item.target, output_dir)
        entries.append(
            {
                "track_id": source_meta.get("track_id", item.source.stem),
                "genre": item.genre,
                "tags": source_meta.get("tags", []),
                "duration_seconds": source_meta.get("duration_seconds"),
                "path": audio_rel,
                "audio_path": audio_rel,
                "audio_url": source_meta.get("audio_url"),
                "source_file": rel(item.source, project_root),
                "delivery_index": index,
            }
        )
    return {
        "schema_version": "1.0",
        "catalog_id": output_dir.name,
        "catalog_name": output_dir.name,
        "source": "jamendo_delivery_subset",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
        "total_tracks": len(entries),
        "audio_root": "audio",
        "path": "catalog.json",
        "entries": entries,
    }


def ensure_gitkeep(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def remove_path(path: Path, project_root: Path, apply: bool) -> int:
    size = directory_size(path) if path.is_dir() else (path.stat().st_size if path.exists() else 0)
    if not path.exists():
        return 0
    print(f"remove {rel(path, project_root)} ({human_size(size)})")
    if apply:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return size


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    jamendo_dir = data_dir / "datasets" / "jamendo"
    source_audio_dir = (
        args.source_audio_dir.resolve()
        if args.source_audio_dir
        else jamendo_dir / "20260521-201328-mtg-jamendo-cdn-training" / "audio"
    )
    output_dir = jamendo_dir / args.output_name
    output_audio_dir = output_dir / "audio"

    if not source_audio_dir.exists() and output_audio_dir.exists():
        source_audio_dir = output_audio_dir
    if not source_audio_dir.exists():
        raise SystemExit(f"Source audio directory not found: {source_audio_dir}")

    before_size = directory_size(project_root)
    keep_files = select_files(source_audio_dir, output_audio_dir, args.keep_per_genre)
    counts: dict[str, int] = {}
    for item in keep_files:
        counts[item.genre] = counts.get(item.genre, 0) + 1

    print("Mode:", "APPLY" if args.apply else "DRY RUN")
    print("Project:", project_root)
    print("Source audio:", source_audio_dir)
    print("Output dataset:", output_dir)
    print("Keep per genre target:", args.keep_per_genre)
    print("Selected counts:", counts)
    print("Selected audio size:", human_size(sum(item.size_bytes for item in keep_files)))
    print("Project size before:", human_size(before_size))

    if not args.apply:
        print("\nNo files were changed. Re-run with --apply to perform cleanup.")
        return 0

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for item in keep_files:
        hardlink_or_copy(item.source, item.target)
    catalog = build_catalog(project_root, output_dir, keep_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    removed_bytes = 0
    runtime_dirs = [
        data_dir / "generated",
        data_dir / "ranked",
        data_dir / "renders",
        data_dir / "jobs",
        data_dir / "projects",
        data_dir / "embeddings",
        data_dir / "models",
        data_dir / "model_cache",
        data_dir / "tmp",
        data_dir / "artifacts",
        data_dir / "tokens",
        data_dir / "generation_plans",
        data_dir / "fusion_comparisons",
        data_dir / "datasets" / "clean",
        data_dir / "datasets" / "logs",
    ]
    for path in runtime_dirs:
        removed_bytes += remove_path(path, project_root, True)

    for child in sorted(jamendo_dir.iterdir()):
        if child.resolve() == output_dir.resolve():
            continue
        removed_bytes += remove_path(child, project_root, True)

    cache_paths = [
        project_root / ".DS_Store",
        project_root / ".pytest_cache",
        project_root / ".ruff_cache",
        project_root / "dump.rdb",
        project_root / "hybrid_music_engine.egg-info",
    ]
    for path in project_root.rglob("__pycache__"):
        cache_paths.append(path)
    for path in cache_paths:
        removed_bytes += remove_path(path, project_root, True)

    keep_dirs = [
        data_dir / "artifacts",
        data_dir / "datasets",
        data_dir / "embeddings",
        data_dir / "fusion_comparisons",
        data_dir / "generated",
        data_dir / "generation_plans",
        data_dir / "jobs",
        data_dir / "models",
        data_dir / "projects",
        data_dir / "ranked",
        data_dir / "renders",
        data_dir / "tmp",
        data_dir / "tokens",
    ]
    for path in keep_dirs:
        ensure_gitkeep(path)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "keep_per_genre": args.keep_per_genre,
        "selected_counts": counts,
        "selected_audio_size_bytes": sum(item.size_bytes for item in keep_files),
        "removed_bytes_estimate": removed_bytes,
        "output_dataset": rel(output_dir, project_root),
        "catalog_path": rel(output_dir / "catalog.json", project_root),
    }
    report_path = project_root / "delivery_cleanup_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    after_size = directory_size(project_root)
    print("\nCleanup complete.")
    print("Report:", report_path)
    print("Project size after:", human_size(after_size))
    print("Approx removed:", human_size(removed_bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
