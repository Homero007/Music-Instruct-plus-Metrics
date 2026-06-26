from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


METADATA_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/"
    "autotagging_genre.tsv"
)
CDN_BASE = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MTG-Jamendo audio by genre tags.")
    parser.add_argument("--tags-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-genre", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    audio_root = output_dir / "audio"
    metadata_root = output_dir / "metadata"
    audio_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    genre_tags = json.loads(Path(args.tags_json).read_text(encoding="utf-8"))
    metadata_text = download_text(METADATA_URL)
    (metadata_root / "autotagging_genre.tsv").write_text(metadata_text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(metadata_text), delimiter="\t"))
    selected = select_rows(rows, genre_tags, args.max_per_genre)

    tasks = [
        (genre, row, audio_root / genre)
        for genre, genre_rows in selected.items()
        for row in genre_rows
    ]
    print(f"output_dir={output_dir}", flush=True)
    print(f"max_per_genre={args.max_per_genre} workers={args.workers}", flush=True)
    print(f"total_tasks={len(tasks)}", flush=True)

    entries: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_entry, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            entry, failure = future.result()
            if entry:
                entries.append(entry)
            if failure:
                failures.append(failure)
            if index % 50 == 0 or index == len(tasks):
                counts = count_by_genre(entries)
                print(
                    f"progress={index}/{len(tasks)} entries={len(entries)} "
                    f"failures={len(failures)} counts={counts}",
                    flush=True,
                )

    catalog = {
        "schema_version": "mtg-jamendo-cdn-catalog-v1",
        "catalog_id": output_dir.name,
        "source": "MTG-Jamendo metadata + MTG/Freesound CDN raw_30s audio",
        "metadata_url": METADATA_URL,
        "cdn_base": CDN_BASE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "genre_tags": genre_tags,
        "max_tracks_per_genre": args.max_per_genre,
        "counts": count_by_genre(entries),
        "total_tracks": len(entries),
        "audio_root": str(audio_root),
        "path": str(output_dir / "catalog.json"),
        "entries": sorted(entries, key=lambda item: (item["genre"], item["track_id"])),
        "failures": failures,
    }
    (output_dir / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"catalog={output_dir / 'catalog.json'}", flush=True)
    print(f"done total={len(entries)} failures={len(failures)}", flush=True)


def download_text(url: str) -> str:
    with urlopen(Request(url, headers={"User-Agent": "hybrid-music-engine/0.1"}), timeout=60) as response:
        return response.read().decode("utf-8")


def select_rows(
    rows: list[dict[str, str]],
    genre_tags: dict[str, list[str]],
    max_per_genre: int,
) -> dict[str, list[dict[str, str]]]:
    selected = {genre: [] for genre in genre_tags}
    seen = {genre: set() for genre in genre_tags}
    normalized = {
        genre: {tag.lower().strip() for tag in tags}
        for genre, tags in genre_tags.items()
    }
    for row in rows:
        row_tags = {
            tag.replace("genre---", "").strip().lower()
            for tag in row.get("TAGS", "").split(",")
            if tag.strip()
        }
        track_id = row.get("TRACK_ID", "")
        for genre, tags in normalized.items():
            if len(selected[genre]) >= max_per_genre:
                continue
            if track_id and track_id not in seen[genre] and row_tags & tags:
                selected[genre].append(row)
                seen[genre].add(track_id)
    return selected


def download_entry(task: tuple[str, dict[str, str], Path]) -> tuple[dict | None, dict | None]:
    genre, row, genre_dir = task
    genre_dir.mkdir(parents=True, exist_ok=True)
    track_id = row["TRACK_ID"]
    track_num = track_id.replace("track_", "").lstrip("0") or track_id
    source_url = f"{CDN_BASE}/{row['PATH']}"
    target = genre_dir / f"{track_num}.mp3"
    try:
        if not target.exists() or target.stat().st_size == 0:
            temp = target.with_suffix(".mp3.part")
            with urlopen(
                Request(source_url, headers={"User-Agent": "hybrid-music-engine/0.1"}),
                timeout=90,
            ) as response:
                with temp.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temp.replace(target)
        return {
            "track_id": track_id,
            "genre": genre,
            "tags": [tag for tag in row.get("TAGS", "").split(",") if tag],
            "duration_seconds": safe_float(row.get("DURATION")),
            "path": row.get("PATH"),
            "audio_url": source_url,
            "audio_path": str(target.resolve()),
        }, None
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        target.with_suffix(".mp3.part").unlink(missing_ok=True)
        return None, {"genre": genre, "track_id": track_id, "url": source_url, "error": str(exc)}


def count_by_genre(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["genre"]] = counts.get(entry["genre"], 0) + 1
    return counts


def safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
