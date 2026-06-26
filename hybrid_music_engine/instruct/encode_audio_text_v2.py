#!/usr/bin/env python3
"""
encode_audio_text_v2.py — Sucesor de encode_audio_text.py.

Diferencias clave frente al original:
  • T5 guarda la SECUENCIA COMPLETA (L, 768) por texto, no un vector pooled.
  • Los prompts se construyen como CAPTIONS RICOS multi-campo (caption_builder).
  • Soporta un manifiesto de INSTRUCCIONES de edición para fine-tuning del editor.
  • EnCodec por defecto compatible con MusicGen (32 kHz, 4 codebooks).

Estructura de salida en data/encodings_v2/:
  encodings_v2/
  ├── t5_seq/
  │   ├── genre__classical.npz        ← (L, 768) secuencia del caption del género
  │   ├── genre__electronic.npz
  │   ├── track__<key>.npz            ← (L, 768) por pista (si hay manifiesto)
  │   └── edit__<key>.npz             ← (L, 768) por instrucción (si hay edits)
  └── encodec/
      └── <rel_path>/
          ├── <stem>_codes.npy        ← (n_codebooks, n_frames)
          └── <stem>_embed.npy        ← (n_frames, dim)  ← memoria de fusión

Uso:
    # T5 (captions ricos por género) + EnCodec sobre segmentos
    python encode_audio_text_v2.py --segments data/segments

    # Solo T5 desde un manifiesto de pistas con metadatos
    python encode_audio_text_v2.py --mode t5 --tracks data/manifests/tracks.csv

    # Codificar instrucciones de edición para entrenar el editor
    python encode_audio_text_v2.py --mode t5 --edits data/manifests/edits.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

# Permite ejecutar como script suelto o como módulo del paquete.
try:
    from . import caption_builder as cb
    from .audio_encoder import DEFAULT_ENCODEC, EnCodecAudioEncoder
    from .text_encoder import T5SequenceEncoder, save_sequence
except ImportError:  # ejecución directa
    import caption_builder as cb  # type: ignore
    from audio_encoder import DEFAULT_ENCODEC, EnCodecAudioEncoder  # type: ignore
    from text_encoder import T5SequenceEncoder, save_sequence  # type: ignore

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def _sanitize(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]+", "_", text).strip("_") or "x"


# ── Lectura de manifiestos ────────────────────────────────────────────────────

def read_tracks(path: Path) -> list[tuple[str, str]]:
    """
    Lee un manifiesto de pistas (CSV o JSON) y devuelve [(clave, caption_rico)].

    Columnas/campos reconocidos: key|id|audio, genre, instruments, tempo_bpm,
    key|tonalidad, mood, style, energy. Construye el caption con caption_builder.
    """
    rows = _read_rows(path)
    id_columns = {"id", "track_id", "audio", "name", "path", "audio_path"}
    items: list[tuple[str, str]] = []
    for row in rows:
        ident = (row.get("id") or row.get("track_id") or row.get("audio")
                 or row.get("name") or "").strip()
        if not ident:
            continue
        meta_row = {k: v for k, v in row.items() if k not in id_columns}
        meta = cb.TrackMeta.from_dict(meta_row)
        caption = cb.build_caption(meta, language="en")
        items.append((f"track__{_sanitize(ident)}", caption))
    return items


def read_edits(path: Path) -> list[tuple[str, str]]:
    """
    Lee un manifiesto de instrucciones de edición y devuelve [(clave, instrucción)].

    Acepta dos formatos:
      • columna `instruction` con el texto literal, o
      • columnas `action`,`target`[,`replacement`,`keep`] para construirlo.
    """
    rows = _read_rows(path)
    items: list[tuple[str, str]] = []
    for i, row in enumerate(rows):
        ident = (row.get("key") or row.get("id") or str(i)).strip()
        literal = (row.get("instruction") or row.get("text") or "").strip()
        if literal:
            instruction = literal
        else:
            action = (row.get("action") or "").strip()
            target = (row.get("target") or "").strip()
            if not action or not target:
                continue
            instruction = cb.build_instruction(
                action, target,
                replacement=row.get("replacement"),
                keep=row.get("keep"),
            )
        items.append((f"edit__{_sanitize(ident)}", instruction))
    return items


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("entries", payload.get("rows", []))
        return [dict(r) for r in payload]
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


# ── Pipeline T5 (secuencias completas) ────────────────────────────────────────

def encode_t5(
    genres: list[str],
    output_dir: Path,
    tracks_csv: Path | None = None,
    edits_csv: Path | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    encoder = T5SequenceEncoder(device=device)
    t5_dir = output_dir / "t5_seq"
    t5_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    items: list[tuple[str, str]] = []
    # 1) Captions ricos por género (fallback siempre disponible)
    for genre in genres:
        caption = cb.default_genre_caption(genre, language="en")
        items.append((f"genre__{_sanitize(genre)}", caption))

    # 2) Captions por pista
    if tracks_csv and tracks_csv.exists():
        track_items = read_tracks(tracks_csv)
        log.info("Manifiesto de pistas: %d entradas", len(track_items))
        items += track_items

    # 3) Instrucciones de edición
    if edits_csv and edits_csv.exists():
        edit_items = read_edits(edits_csv)
        log.info("Manifiesto de ediciones: %d entradas", len(edit_items))
        items += edit_items

    log.info("Codificando %d textos con T5 (secuencia completa)…", len(items))
    for key, text in items:
        enc = encoder.encode_sequence(text)
        hidden, _ = enc.single()                 # (L, 768) sin padding
        path = t5_dir / f"{key}.npz"
        save_sequence(path, hidden, text)
        saved[key] = path
        log.info("  [%-28s] L=%-3d  «%s»", key, hidden.shape[0], text[:54])

    return saved


# ── Pipeline EnCodec ──────────────────────────────────────────────────────────

def encode_encodec(
    segments_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_ENCODEC,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, int]:
    encoder = EnCodecAudioEncoder(model_name=model_name, device=device)
    encodec_dir = output_dir / "encodec"
    counts: dict[str, int] = {}
    errors: list[str] = []

    audio_files = sorted(
        p for p in segments_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        log.warning("No se encontraron audios en %s", segments_dir)
        return counts

    log.info("EnCodec: %d archivos…", len(audio_files))
    for i, audio_path in enumerate(audio_files, 1):
        try:
            rel = audio_path.relative_to(segments_dir)
        except ValueError:
            rel = Path(audio_path.name)
        out_dir = encodec_dir / rel.parent
        codes_path = out_dir / f"{audio_path.stem}_codes.npy"
        embed_path = out_dir / f"{audio_path.stem}_embed.npy"

        key = str(rel.parent)
        if not overwrite and codes_path.exists() and embed_path.exists():
            counts[key] = counts.get(key, 0) + 1
            continue

        log.info("[%d/%d] %s", i, len(audio_files), rel)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            result = encoder.encode_file(audio_path)
            np.save(codes_path, result["codes"].astype(np.int16))
            np.save(embed_path, result["embeddings"].astype(np.float32))
            counts[key] = counts.get(key, 0) + 1
        except Exception as exc:  # noqa: BLE001 — registramos y seguimos
            log.warning("  Error en %s: %s — omitido", audio_path.name, exc)
            errors.append(str(audio_path))

    if errors:
        log.warning("%d archivos omitidos por error.", len(errors))
    log.info("EnCodec completado. Total: %d", sum(counts.values()))
    return counts


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Codificador T5 (secuencia completa) + EnCodec para Instruct-MusicGen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["all", "t5", "encodec"], default="all")
    p.add_argument("--segments", type=Path, default=Path("data/segments"))
    p.add_argument("--tracks", type=Path, default=None,
                   help="CSV/JSON con metadatos por pista para captions ricos")
    p.add_argument("--edits", type=Path, default=None,
                   help="CSV/JSON con instrucciones de edición")
    p.add_argument("--genres", nargs="+",
                   default=["classical", "electronic", "reggaeton"])
    p.add_argument("--output-dir", type=Path, default=Path("data/encodings_v2"))
    p.add_argument("--encodec-model", default=DEFAULT_ENCODEC,
                   help="Modelo EnCodec (32khz=compatible MusicGen, 24khz=experimental)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode in ("all", "t5"):
        log.info("PASO 1/2 — T5 (secuencia completa, captions ricos)")
        encode_t5(
            genres=args.genres,
            output_dir=args.output_dir,
            tracks_csv=args.tracks,
            edits_csv=args.edits,
            device=args.device,
        )

    if args.mode in ("all", "encodec"):
        log.info("PASO 2/2 — EnCodec (codes RVQ + embedding de fusión)")
        if not args.segments.exists():
            log.error("No existe --segments: %s", args.segments)
            sys.exit(1)
        counts = encode_encodec(
            segments_dir=args.segments,
            output_dir=args.output_dir,
            model_name=args.encodec_model,
            device=args.device,
            overwrite=args.overwrite,
        )
        print("\n  EnCodec — archivos por bloque/género:")
        for key, n in sorted(counts.items()):
            print(f"    {key:<32} {n:>5} clips")

    print(f"\n  Encodings en: {args.output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
