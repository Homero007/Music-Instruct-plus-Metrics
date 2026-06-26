from __future__ import annotations

import json
import shutil
import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id


JAMENDO_API = "https://api.jamendo.com/v3.0/tracks"
JAMENDO_CLIENT_ID = "b6747d04"

JAMENDO_GENRE_TAGS: dict[str, list[str]] = {
    "classical": ["classical", "orchestral", "piano", "strings"],
    "electronic": ["electronic", "techno", "house", "ambient", "edm", "trance"],
    "reggaeton": ["reggae", "latin", "reggaeton"],
}


def download_jamendo_catalog(
    config: EngineConfig,
    *,
    genre_tags: dict[str, list[str]] | None = None,
    client_id: str = JAMENDO_CLIENT_ID,
    catalog_name: str = "mtg_jamendo",
    tracks_per_page: int = 200,
    max_tracks_per_genre: int | None = 500,
    download_audio: bool = True,
    api_url: str = JAMENDO_API,
    source: str = "mtg-cdn",
    concurrent_downloads: int = 16,
) -> dict[str, Any]:
    if source in {"mtg-cdn", "mtg", "cdn"}:
        return download_mtg_jamendo_catalog(
            config,
            genre_tags=genre_tags,
            catalog_name=catalog_name,
            max_tracks_per_genre=max_tracks_per_genre or 500,
            download_audio=download_audio,
            concurrent_downloads=concurrent_downloads,
        )
    if source not in {"api", "jamendo-api"}:
        raise RuntimeError("source debe ser 'mtg-cdn' o 'api'.")

    normalized_tags = _normalize_genre_tags(genre_tags or JAMENDO_GENRE_TAGS)
    if tracks_per_page <= 0:
        raise RuntimeError("tracks_per_page debe ser mayor que cero.")
    if max_tracks_per_genre is not None and max_tracks_per_genre <= 0:
        raise RuntimeError("max_tracks_per_genre debe ser mayor que cero o null.")

    catalog_id = create_id(catalog_name, prefix="jamendo")
    root = config.datasets_dir / "jamendo" / catalog_id
    audio_root = root / "audio"
    metadata_root = root / "metadata"
    audio_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for genre, tags in normalized_tags.items():
        genre_count = 0
        offset = 0
        genre_dir = audio_root / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        while True:
            if max_tracks_per_genre is not None and genre_count >= max_tracks_per_genre:
                break
            page_limit = tracks_per_page
            if max_tracks_per_genre is not None:
                page_limit = min(page_limit, max_tracks_per_genre - genre_count)
            page = query_jamendo_tracks(
                api_url=api_url,
                client_id=client_id,
                tags=tags,
                limit=page_limit,
                offset=offset,
            )
            results = page.get("results", [])
            if not results:
                break
            (metadata_root / f"{genre}_{offset}.json").write_text(
                json.dumps(page, indent=2),
                encoding="utf-8",
            )
            for track in results:
                track_id = str(track.get("id") or "")
                if not track_id:
                    rejected.append({"genre": genre, "reason": "missing_track_id", "track": track})
                    continue
                audio_url = track.get("audiodownload") or track.get("audio")
                audio_path: str | None = None
                download_error: str | None = None
                if download_audio:
                    if not audio_url:
                        download_error = "missing_audio_url"
                    else:
                        try:
                            target_path = genre_dir / f"{track_id}.mp3"
                            download_file(str(audio_url), target_path)
                            audio_path = str(target_path)
                        except RuntimeError as exc:
                            download_error = str(exc)
                entries.append(
                    {
                        "track_id": track_id,
                        "genre": genre,
                        "tags": tags,
                        "name": track.get("name"),
                        "artist_name": track.get("artist_name"),
                        "album_name": track.get("album_name"),
                        "duration_seconds": _safe_float(track.get("duration")),
                        "license_ccurl": track.get("license_ccurl"),
                        "audio_url": audio_url,
                        "audio_path": audio_path,
                        "download_error": download_error,
                        "raw": track,
                    }
                )
                genre_count += 1
                if max_tracks_per_genre is not None and genre_count >= max_tracks_per_genre:
                    break
            offset += len(results)
        counts[genre] = genre_count

    catalog_path = root / "catalog.json"
    catalog = {
        "schema_version": "jamendo-audio-catalog-v1",
        "catalog_id": catalog_id,
        "catalog_name": catalog_name,
        "source": "MTG-Jamendo / Jamendo API",
        "api_url": api_url,
        "client_id": client_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "genre_tags": normalized_tags,
        "genre_count": len(normalized_tags),
        "tracks_per_page": tracks_per_page,
        "max_tracks_per_genre": max_tracks_per_genre,
        "download_audio": download_audio,
        "counts": counts,
        "total_tracks": len(entries),
        "root": str(root),
        "audio_root": str(audio_root),
        "path": str(catalog_path),
        "entries": entries,
        "rejected": rejected,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def download_mtg_jamendo_catalog(
    config: EngineConfig,
    *,
    genre_tags: dict[str, list[str]] | None = None,
    catalog_name: str = "mtg_jamendo",
    max_tracks_per_genre: int = 500,
    download_audio: bool = True,
    concurrent_downloads: int = 16,
    metadata_url: str = (
        "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/"
        "autotagging_genre.tsv"
    ),
    cdn_base: str = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio",
) -> dict[str, Any]:
    normalized_tags = _normalize_genre_tags(genre_tags or JAMENDO_GENRE_TAGS)
    if max_tracks_per_genre <= 0:
        raise RuntimeError("max_tracks_per_genre debe ser mayor que cero.")
    if concurrent_downloads <= 0:
        raise RuntimeError("concurrent_downloads debe ser mayor que cero.")

    catalog_id = create_id(catalog_name, prefix="jamendo")
    root = config.datasets_dir / "jamendo" / catalog_id
    audio_root = root / "audio"
    metadata_root = root / "metadata"
    audio_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    metadata_text = _download_text(metadata_url)
    metadata_file = metadata_root / "autotagging_genre.tsv"
    metadata_file.write_text(metadata_text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(metadata_text), delimiter="\t"))
    selected = _select_mtg_rows(rows, normalized_tags, max_tracks_per_genre)

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tasks = [
        (genre, row, cdn_base, audio_root / genre, download_audio)
        for genre, genre_rows in selected.items()
        for row in genre_rows
    ]

    with ThreadPoolExecutor(max_workers=concurrent_downloads) as pool:
        futures = [pool.submit(_materialize_mtg_entry, task) for task in tasks]
        for future in as_completed(futures):
            entry, failure = future.result()
            if entry:
                entries.append(entry)
            if failure:
                failures.append(failure)

    counts = {
        genre: sum(1 for entry in entries if entry["genre"] == genre)
        for genre in normalized_tags
    }
    catalog_path = root / "catalog.json"
    catalog = {
        "schema_version": "mtg-jamendo-cdn-catalog-v1",
        "catalog_id": catalog_id,
        "catalog_name": catalog_name,
        "source": "MTG-Jamendo metadata + MTG/Freesound CDN raw_30s audio",
        "metadata_url": metadata_url,
        "cdn_base": cdn_base,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "genre_tags": normalized_tags,
        "genre_count": len(normalized_tags),
        "max_tracks_per_genre": max_tracks_per_genre,
        "download_audio": download_audio,
        "concurrent_downloads": concurrent_downloads,
        "counts": counts,
        "total_tracks": len(entries),
        "root": str(root),
        "audio_root": str(audio_root),
        "path": str(catalog_path),
        "metadata_file": str(metadata_file),
        "entries": sorted(entries, key=lambda item: (item["genre"], item["track_id"])),
        "failures": failures,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def list_jamendo_catalogs(config: EngineConfig) -> list[dict[str, Any]]:
    catalogs: list[dict[str, Any]] = []
    for path in sorted((config.datasets_dir / "jamendo").glob("**/catalog.json")):
        payload = _read_catalog(path)
        if not payload:
            continue
        counts = _catalog_counts(payload)
        total_tracks = int(payload.get("total_tracks") or len(payload.get("entries", [])))
        if total_tracks <= 0:
            continue
        catalogs.append(
            {
                "catalog_id": payload.get("catalog_id") or path.parent.name,
                "catalog_name": payload.get("catalog_name") or path.parent.name,
                "path": str(path),
                "created_at": payload.get("created_at"),
                "source": payload.get("source"),
                "total_tracks": total_tracks,
                "counts": counts,
                "genres": sorted(counts.keys()),
                "audio_root": payload.get("audio_root"),
            }
        )
    return sorted(catalogs, key=lambda item: int(item.get("total_tracks") or 0), reverse=True)


def select_jamendo_catalog_entries(
    config: EngineConfig,
    *,
    catalog_path: Path,
    genres: list[str],
    max_tracks_per_genre: int,
    output_name: str = "selected_jamendo",
) -> dict[str, Any]:
    if max_tracks_per_genre <= 0:
        raise RuntimeError("max_tracks_per_genre debe ser mayor que cero.")
    selected_genres = [genre.strip().lower() for genre in genres if genre.strip()]
    if not selected_genres:
        raise RuntimeError("Selecciona al menos un género.")
    source_path = catalog_path.expanduser().resolve()
    catalog = _read_catalog(source_path)
    if not catalog:
        raise RuntimeError(f"No se pudo leer el catálogo Jamendo: {source_path}")

    available = _catalog_counts(catalog)
    missing = [genre for genre in selected_genres if genre not in available]
    if missing:
        raise RuntimeError(f"Géneros no disponibles en el catálogo: {', '.join(missing)}")

    grouped: dict[str, list[dict[str, Any]]] = {genre: [] for genre in selected_genres}
    rejected: list[dict[str, Any]] = []
    for entry in catalog.get("entries", []):
        genre = str(entry.get("genre", "")).strip().lower()
        if genre in grouped and len(grouped[genre]) < max_tracks_per_genre:
            normalized_entry = _normalized_selectable_audio_entry(
                entry,
                config=config,
                catalog_file=source_path,
            )
            if normalized_entry:
                grouped[genre].append(normalized_entry)
            else:
                rejected.append(
                    {
                        "track_id": entry.get("track_id"),
                        "genre": genre,
                        "audio_path": entry.get("audio_path"),
                        "reason": "audio_file_not_found_or_missing",
                    }
                )

    entries = [entry for genre in selected_genres for entry in grouped[genre]]
    counts = {genre: len(grouped[genre]) for genre in selected_genres}
    if not entries:
        raise RuntimeError(
            "No hay audios locales utilizables para los géneros seleccionados. "
            "Revisa que el catálogo tenga audio_path válido o descarga audio primero."
        )
    output_id = create_id(output_name, prefix="jamendo-selection")
    output_dir = config.datasets_dir / "jamendo" / "selections" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "catalog.json"
    selected_catalog = {
        "schema_version": "mtg-jamendo-selection-v1",
        "catalog_id": output_id,
        "catalog_name": output_name,
        "source": "Filtered MTG-Jamendo catalog",
        "source_catalog_path": str(source_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_genres": selected_genres,
        "max_tracks_per_genre": max_tracks_per_genre,
        "counts": counts,
        "total_tracks": len(entries),
        "audio_root": catalog.get("audio_root"),
        "path": str(output_path),
        "entries": entries,
        "rejected": rejected,
    }
    output_path.write_text(json.dumps(selected_catalog, indent=2), encoding="utf-8")
    return selected_catalog


def prepare_jamendo_clips(
    config: EngineConfig,
    *,
    catalog_path: Path,
    clip_duration_seconds: float = 20.0,
    hop_duration_seconds: float | None = None,
    max_clips_per_track: int | None = None,
    min_clip_seconds: float = 5.0,
    sample_rate: int | None = None,
    mono: bool = True,
) -> dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Preparar clips requiere soundfile. Instala: python -m pip install -e '.[audio]'"
        ) from exc

    if clip_duration_seconds <= 0:
        raise RuntimeError("clip_duration_seconds debe ser mayor que cero.")
    if hop_duration_seconds is not None and hop_duration_seconds <= 0:
        raise RuntimeError("hop_duration_seconds debe ser mayor que cero o null.")
    if max_clips_per_track is not None and max_clips_per_track <= 0:
        raise RuntimeError("max_clips_per_track debe ser mayor que cero o null.")
    if min_clip_seconds <= 0:
        raise RuntimeError("min_clip_seconds debe ser mayor que cero.")

    catalog_file = Path(catalog_path).expanduser().resolve()
    if not catalog_file.exists():
        raise RuntimeError(f"Catálogo Jamendo no encontrado: {catalog_file}")
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    catalog_id = catalog.get("catalog_id") or create_id(catalog_file.stem, prefix="jamendo")
    output_root = config.datasets_dir / "jamendo" / str(catalog_id) / "clips"
    output_root.mkdir(parents=True, exist_ok=True)

    sr = sample_rate or config.default_sample_rate
    hop = hop_duration_seconds or clip_duration_seconds
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for track in catalog.get("entries", []):
        audio_path_value = track.get("audio_path")
        genre = str(track.get("genre") or "unknown")
        track_id = str(track.get("track_id") or create_id("track", prefix="track"))
        if not audio_path_value:
            rejected.append(
                {
                    "track_id": track_id,
                    "genre": genre,
                    "reason": "missing_audio_path",
                }
            )
            continue
        source = _resolve_audio_path(str(audio_path_value), config=config, catalog_file=catalog_file)
        if not source.exists():
            rejected.append(
                {
                    "track_id": track_id,
                    "genre": genre,
                    "source_audio": str(source),
                    "reason": "audio_file_not_found",
                }
            )
            continue
        try:
            audio, loaded_sr = _load_clip_audio(source, sample_rate=sr, mono=mono)
        except (OSError, ValueError, RuntimeError) as exc:
            rejected.append(
                {
                    "track_id": track_id,
                    "genre": genre,
                    "source_audio": str(source),
                    "reason": f"audio_read_error: {exc}",
                }
            )
            continue
        if audio.ndim > 1:
            total_samples = audio.shape[-1]
        else:
            total_samples = len(audio)
        clip_samples = int(round(clip_duration_seconds * loaded_sr))
        hop_samples = int(round(hop * loaded_sr))
        min_samples = int(round(min_clip_seconds * loaded_sr))
        if total_samples < min_samples:
            rejected.append(
                {
                    "track_id": track_id,
                    "genre": genre,
                    "source_audio": str(source),
                    "duration_seconds": round(total_samples / loaded_sr, 4),
                    "reason": "audio_shorter_than_min_clip",
                }
            )
            continue

        genre_dir = output_root / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        track_clip_count = 0
        start_sample = 0
        while start_sample < total_samples:
            if max_clips_per_track is not None and track_clip_count >= max_clips_per_track:
                break
            end_sample = min(start_sample + clip_samples, total_samples)
            if end_sample - start_sample < min_samples:
                break
            clip = audio[..., start_sample:end_sample] if audio.ndim > 1 else audio[start_sample:end_sample]
            clip_id = create_id(f"{track_id}-{track_clip_count + 1}", prefix="clip")
            clip_path = genre_dir / f"{track_id}_clip_{track_clip_count + 1:03d}.wav"
            sf.write(clip_path, clip.T if getattr(clip, "ndim", 1) > 1 else clip, loaded_sr, subtype="PCM_24")
            duration = round((end_sample - start_sample) / loaded_sr, 4)
            entries.append(
                {
                    "clip_id": clip_id,
                    "track_id": track_id,
                    "genre": genre,
                    "clip_index": track_clip_count + 1,
                    "source_audio": str(source),
                    "clip_path": str(clip_path),
                    "start_seconds": round(start_sample / loaded_sr, 4),
                    "duration_seconds": duration,
                    "sample_rate": int(loaded_sr),
                    "source_track": {
                        "name": track.get("name"),
                        "artist_name": track.get("artist_name"),
                        "license_ccurl": track.get("license_ccurl"),
                    },
                }
            )
            counts[genre] = counts.get(genre, 0) + 1
            track_clip_count += 1
            start_sample += hop_samples

    clip_catalog_path = output_root / "clips_catalog.json"
    clip_catalog = {
        "schema_version": "jamendo-clip-catalog-v1",
        "catalog_id": create_id("jamendo_clips", prefix="clips"),
        "source_catalog_path": str(catalog_file),
        "source_catalog_id": catalog.get("catalog_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "clip_duration_seconds": clip_duration_seconds,
        "hop_duration_seconds": hop,
        "max_clips_per_track": max_clips_per_track,
        "min_clip_seconds": min_clip_seconds,
        "sample_rate": sr,
        "mono": mono,
        "counts": counts,
        "total_clips": len(entries),
        "root": str(output_root),
        "path": str(clip_catalog_path),
        "entries": entries,
        "rejected": rejected,
    }
    clip_catalog_path.write_text(json.dumps(clip_catalog, indent=2), encoding="utf-8")
    if not entries:
        if rejected:
            first = rejected[0]
            raise RuntimeError(
                "No se pudo crear ningún clip. "
                f"Primer rechazo: {first.get('reason')} · {first.get('source_audio') or first.get('track_id')}"
            )
        raise RuntimeError("No se pudo crear ningún clip: el catálogo no contiene audios utilizables.")
    return clip_catalog


def _resolve_audio_path(value: str, *, config: EngineConfig, catalog_file: Path) -> Path:
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([catalog_file.parent / path, config.project_root / path])
        if path.parts and path.parts[0] == config.project_root.name:
            candidates.append(config.project_root.parent / path)

    raw = str(path)
    duplicate_root = str(config.project_root / config.project_root.name)
    if raw.startswith(duplicate_root):
        candidates.append(Path(str(config.project_root) + raw[len(duplicate_root) :]))

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return candidates[0].expanduser().resolve() if candidates else path.resolve()


def _normalized_selectable_audio_entry(
    entry: dict[str, Any],
    *,
    config: EngineConfig,
    catalog_file: Path,
) -> dict[str, Any] | None:
    audio_path = entry.get("audio_path")
    if not audio_path:
        return None
    resolved = _resolve_audio_path(str(audio_path), config=config, catalog_file=catalog_file)
    if not resolved.exists():
        return None
    return {**entry, "audio_path": str(resolved)}


def _load_clip_audio(source: Path, *, sample_rate: int, mono: bool):
    try:
        import numpy as np
        import soundfile as sf

        audio, loaded_sr = sf.read(source, dtype="float32", always_2d=False)
        if loaded_sr != sample_rate:
            raise RuntimeError("sample_rate_mismatch")
        if mono and getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        elif not mono and getattr(audio, "ndim", 1) == 1:
            audio = audio.reshape(-1, 1)
        return audio, loaded_sr
    except (OSError, RuntimeError, ValueError):
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(
                "No se pudo leer/remuestrear audio con soundfile. Para MP3 o resampling instala librosa: "
                "python -m pip install -e '.[audio]'"
            ) from exc
        return librosa.load(source, sr=sample_rate, mono=mono)


def query_jamendo_tracks(
    *,
    api_url: str,
    client_id: str,
    tags: list[str],
    limit: int,
    offset: int,
) -> dict[str, Any]:
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": str(limit),
        "offset": str(offset),
        "include": "musicinfo",
        "audioformat": "mp31",
        "order": "popularity_total",
        "tags": " ".join(tags),
    }
    url = f"{api_url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "hybrid-music-engine/0.1"})
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No se pudo consultar Jamendo: {exc}") from exc


def _download_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "hybrid-music-engine/0.1"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"No se pudo descargar metadata MTG-Jamendo: {exc}") from exc


def _read_catalog(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _catalog_counts(catalog: dict[str, Any]) -> dict[str, int]:
    raw_counts = catalog.get("counts")
    if isinstance(raw_counts, dict) and raw_counts:
        return {
            str(genre).strip().lower(): int(count)
            for genre, count in raw_counts.items()
            if str(genre).strip()
        }
    counts: dict[str, int] = {}
    for entry in catalog.get("entries", []):
        genre = str(entry.get("genre", "")).strip().lower()
        if genre:
            counts[genre] = counts.get(genre, 0) + 1
    return counts


def _select_mtg_rows(
    rows: list[dict[str, str]],
    genre_tags: dict[str, list[str]],
    max_tracks_per_genre: int,
) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = {genre: [] for genre in genre_tags}
    seen: dict[str, set[str]] = {genre: set() for genre in genre_tags}
    normalized_lookup = {
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
        if not track_id:
            continue
        for genre, tags in normalized_lookup.items():
            if len(selected[genre]) >= max_tracks_per_genre:
                continue
            if row_tags & tags and track_id not in seen[genre]:
                selected[genre].append(row)
                seen[genre].add(track_id)
    return selected


def _materialize_mtg_entry(
    task: tuple[str, dict[str, str], str, Path, bool],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    genre, row, cdn_base, genre_dir, download_audio = task
    track_id = row.get("TRACK_ID", "")
    track_num = track_id.replace("track_", "").lstrip("0") or track_id
    source_url = f"{cdn_base}/{row.get('PATH', '')}"
    audio_path: str | None = None
    status = "metadata-only"
    if download_audio:
        try:
            target = genre_dir / f"{track_num}.mp3"
            download_file(source_url, target)
            audio_path = str(target)
            status = "downloaded"
        except RuntimeError as exc:
            return None, {
                "genre": genre,
                "track_id": track_id,
                "url": source_url,
                "error": str(exc),
            }

    return {
        "track_id": track_id,
        "genre": genre,
        "tags": [tag for tag in row.get("TAGS", "").split(",") if tag],
        "duration_seconds": _safe_float(row.get("DURATION")),
        "path": row.get("PATH"),
        "audio_url": source_url,
        "audio_path": audio_path,
        "status": status,
        "artist_id": row.get("ARTIST_ID"),
        "album_id": row.get("ALBUM_ID"),
    }, None


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        return
    request = Request(url, headers={"User-Agent": "hybrid-music-engine/0.1"})
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    try:
        with urlopen(request, timeout=120) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        temp_path.replace(target_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No se pudo descargar audio: {exc}") from exc


def _normalize_genre_tags(genre_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for genre, tags in genre_tags.items():
        genre_name = genre.strip().lower().replace(" ", "_")
        clean_tags = [tag.strip().lower() for tag in tags if tag.strip()]
        if not genre_name:
            raise RuntimeError("Hay un género sin nombre.")
        if not clean_tags:
            raise RuntimeError(f"El género '{genre_name}' no tiene tags.")
        if genre_name in normalized:
            raise RuntimeError(f"Género duplicado: {genre_name}")
        normalized[genre_name] = clean_tags
    if not normalized:
        raise RuntimeError("Debes indicar al menos un género/tag.")
    return normalized


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
