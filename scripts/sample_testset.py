#!/usr/bin/env python3
"""sample_testset.py — Construcción reproducible del banco de pruebas (MusicCaps).

Construye un banco de 100 clips de audio (10 géneros × 10 clips) a partir de
MusicCaps [Agostinelli et al., 2023], un subconjunto de AudioSet con 5,521
clips de YouTube de 10 s anotados con captions detallados por músicos
profesionales.

Pipeline (semilla fija = 42, reproducible):
  1. Descarga el CSV oficial (musiccaps-public.csv) o usa una copia local.
  2. Filtra captions: >= 20 palabras y que mencionen género + instrumento.
  3. Estratifica por género usando aspect_list (10 géneros objetivo).
  4. Selecciona 10 clips por género (barajado determinista, semilla 42).
  5. Verifica disponibilidad en YouTube con `yt-dlp --simulate` (perezoso:
     se prueban candidatos barajados hasta lograr 10 accesibles por género).
  6. Descarga el tramo de 10 s con yt-dlp y lo recorta con ffmpeg (-ss/-t).
  7. Normaliza a 32 kHz mono con torchaudio.transforms.Resample.
  8. Verifica integridad: cada .wav debe tener EXACTAMENTE 320,000 muestras.
  9. Estima el tempo (BPM) del audio con librosa y lo guarda en metadata.

Nota sobre el tempo (BPM):
  Los captions de MusicCaps describen el tempo de forma CUALITATIVA
  ("fast tempo", "slow groove"); solo 1 de los 5,521 clips trae un BPM
  numérico explícito. Por eso el criterio "tempo en BPM" se satisface
  estimando el BPM del audio con librosa (columna `tempo_bpm`, marcada como
  estimada), y el filtro de caption exige una MENCIÓN de tempo (cualitativa
  o numérica), no un número exacto.

Entregable:
  testset/                       — 100 archivos NNNN_genre_BPM.wav
  testset_metadata.csv           — id, ytid, caption, genre, tempo_bpm, instruments

Uso:
  python scripts/sample_testset.py                 # pipeline completo
  python scripts/sample_testset.py --dry-run       # selección sin descargar
  python scripts/sample_testset.py --csv ruta.csv  # usar CSV local
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

SEED = 42
TARGET_SR = 32_000
CLIP_SECONDS = 10
EXPECTED_SAMPLES = TARGET_SR * CLIP_SECONDS  # 320,000
PER_GENRE = 10
MIN_CAPTION_WORDS = 20
MIN_BPM_PER_GENRE = 2

# Espejos del CSV oficial de MusicCaps (se prueban en orden).
CSV_MIRRORS = [
    "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv",
    "https://raw.githubusercontent.com/LixiangZhao98/MusicCaps/main/musiccaps-public.csv",
    "https://storage.googleapis.com/gresearch/musiccaps/musiccaps-public.csv",
]

# 10 géneros objetivo → palabras clave (en inglés, como aparecen en MusicCaps).
# El primer término de cada lista es el slug usado en los nombres de archivo.
GENRE_KEYWORDS: dict[str, list[str]] = {
    "jazz":       ["jazz", "swing", "bebop", "big band"],
    "rock":       ["rock", "punk", "grunge", "indie rock"],
    "electronic": ["electronic", "techno", "house", "edm", "trance", "synth", "electronica"],
    "classical":  ["classical", "orchestral", "symphony", "baroque", "chamber music"],
    "pop":        ["pop ", "pop,", "k-pop", "synth-pop", "dance pop"],
    "hiphop":     ["hip hop", "hip-hop", "rap", "trap", "boom bap"],
    "rnb":        ["r&b", "rnb", "rhythm and blues", "soul", "neo soul"],
    "metal":      ["metal", "heavy metal", "death metal", "thrash"],
    "folk":       ["folk", "acoustic folk", "bluegrass", "americana"],
    "latin":      ["latin", "salsa", "reggaeton", "bossa", "cumbia", "merengue", "bachata"],
}

# Instrumentos reconocidos (para la columna `instruments` y el filtro de caption).
INSTRUMENT_TERMS = [
    "guitar", "electric guitar", "acoustic guitar", "bass", "drums", "drum machine",
    "piano", "keyboard", "synth", "synthesizer", "violin", "cello", "strings",
    "trumpet", "saxophone", "sax", "flute", "clarinet", "trombone", "organ",
    "vocals", "voice", "choir", "harmonica", "banjo", "ukulele", "harp",
    "percussion", "congas", "bongos", "tabla", "accordion", "horns", "brass",
]

BPM_RE = re.compile(r"(\d{2,3})\s*(?:bpm|beats per minute)", re.IGNORECASE)


# ── CSV de MusicCaps ──────────────────────────────────────────────────────────

def fetch_csv(out_csv: Path) -> Path:
    if out_csv.exists():
        print(f"  · CSV ya presente: {out_csv}")
        return out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    for url in CSV_MIRRORS:
        try:
            print(f"  · Descargando MusicCaps CSV desde {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if b"ytid" in data[:200]:
                out_csv.write_bytes(data)
                print(f"  · Guardado {out_csv} ({len(data)//1024} KB)")
                return out_csv
        except Exception as exc:  # noqa: BLE401
            print(f"    ! falló: {exc}")
    raise RuntimeError(
        "No se pudo descargar musiccaps-public.csv desde ningún espejo. "
        "Descárgalo manualmente y pásalo con --csv."
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── Anotación: género, tempo, instrumentos ────────────────────────────────────

def detect_genre(text: str) -> str | None:
    low = f" {text.lower()} "
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return genre
    return None


def extract_bpm(text: str) -> int | None:
    match = BPM_RE.search(text)
    if not match:
        return None
    bpm = int(match.group(1))
    return bpm if 30 <= bpm <= 250 else None


def extract_instruments(text: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for term in INSTRUMENT_TERMS:
        if term in low and term not in found:
            found.append(term)
    return found


def mentions_tempo(text: str) -> bool:
    low = text.lower()
    if extract_bpm(text) is not None:
        return True
    return any(w in low for w in (
        "tempo", "fast", "slow", "upbeat", "mid-tempo", "moderate",
        "andante", "allegro", "groove", "rhythm", "beat",
    ))


def annotate(row: dict[str, str]) -> dict[str, Any] | None:
    caption = (row.get("caption") or "").strip()
    aspects = (row.get("aspect_list") or "").strip()
    blob = f"{aspects} {caption}"
    if len(caption.split()) < MIN_CAPTION_WORDS:
        return None
    genre = detect_genre(blob)
    if genre is None:
        return None
    instruments = extract_instruments(blob)
    if not instruments:
        return None
    if not mentions_tempo(blob):
        return None
    return {
        "ytid": (row.get("ytid") or "").strip(),
        "start_s": int(float(row.get("start_s") or 0)),
        "end_s": int(float(row.get("end_s") or 10)),
        "caption": caption,
        "genre": genre,
        "tempo_bpm": extract_bpm(blob),
        "instruments": instruments,
    }


# ── Selección estratificada (seed=42) ─────────────────────────────────────────

def stratify(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {g: [] for g in GENRE_KEYWORDS}
    for row in rows:
        annotated = annotate(row)
        if annotated:
            buckets[annotated["genre"]].append(annotated)
    return buckets


def order_candidates(candidates: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Baraja de forma determinista, priorizando clips con BPM explícito al
    frente para garantizar >= MIN_BPM_PER_GENRE cuando existan."""
    with_bpm = [c for c in candidates if c["tempo_bpm"] is not None]
    without = [c for c in candidates if c["tempo_bpm"] is None]
    rng.shuffle(with_bpm)
    rng.shuffle(without)
    # Coloca primero los necesarios con BPM, luego intercala el resto.
    head = with_bpm[:MIN_BPM_PER_GENRE]
    tail = with_bpm[MIN_BPM_PER_GENRE:] + without
    rng.shuffle(tail)
    return head + tail


# ── Disponibilidad y descarga (yt-dlp + ffmpeg + torchaudio) ──────────────────

def ytdlp_cmd() -> list[str]:
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    return [sys.executable, "-m", "yt_dlp"]


def is_available(ytid: str) -> bool:
    cmd = ytdlp_cmd() + ["--simulate", "--no-warnings", "--quiet",
                         f"https://www.youtube.com/watch?v={ytid}"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def download_clip(item: dict[str, Any], raw_path: Path) -> bool:
    """Descarga el tramo [start, start+10] como WAV usando yt-dlp + ffmpeg."""
    start = item["start_s"]
    end = start + CLIP_SECONDS
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ytdlp_cmd() + [
        "-f", "bestaudio",
        "--extract-audio", "--audio-format", "wav",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--no-warnings", "--quiet",
        "-o", str(raw_path.with_suffix(".%(ext)s")),
        f"https://www.youtube.com/watch?v={item['ytid']}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode == 0 and raw_path.exists()
    except Exception:
        return False


def normalize_to_32k_mono(raw_path: Path, final_path: Path) -> bool:
    """Resamplea a 32 kHz mono y fuerza EXACTAMENTE 320,000 muestras."""
    import torch
    import torchaudio

    try:
        waveform, sr = torchaudio.load(str(raw_path))
    except Exception as exc:
        print(f"      ! no se pudo leer {raw_path.name}: {exc}")
        return False
    # Mono: promedia canales.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    # Fuerza longitud exacta (pad con ceros o trunca).
    n = waveform.shape[1]
    if n < EXPECTED_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, EXPECTED_SAMPLES - n))
    elif n > EXPECTED_SAMPLES:
        waveform = waveform[:, :EXPECTED_SAMPLES]
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(final_path), waveform, TARGET_SR)
    return True


def verify_samples(path: Path) -> bool:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames == EXPECTED_SAMPLES and info.samplerate == TARGET_SR
    except Exception:
        return False


def estimate_bpm(path: Path) -> int | None:
    """Estima el tempo (BPM) del clip con librosa. MusicCaps casi nunca trae
    BPM numérico, así que se deriva del audio (valor estimado, no anotado)."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
        tempo = librosa.feature.tempo(y=y, sr=sr)
        bpm = int(round(float(np.atleast_1d(tempo)[0])))
        return bpm if 30 <= bpm <= 250 else None
    except Exception as exc:  # noqa: BLE001
        print(f"      ! no se pudo estimar BPM de {path.name}: {exc}")
        return None


# ── Orquestación ──────────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    root = Path(__file__).parent.parent
    csv_path = Path(args.csv) if args.csv else root / "data" / "datasets" / "musiccaps-public.csv"
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    meta_path = root / "testset_metadata.csv"
    raw_dir = out_dir / "_raw"

    print("1) Obteniendo CSV de MusicCaps")
    csv_path = fetch_csv(csv_path)
    rows = read_rows(csv_path)
    print(f"   {len(rows)} clips en el CSV")

    print("2-3) Filtrando captions y estratificando por género")
    buckets = stratify(rows)
    for genre, items in buckets.items():
        print(f"   {genre:<11} {len(items)} candidatos válidos")

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    index = 0

    print(f"\n4-8) Seleccionando, verificando y descargando {PER_GENRE}/género")
    for genre in GENRE_KEYWORDS:
        ordered = order_candidates(buckets[genre], rng)
        kept = 0
        for item in ordered:
            if kept >= PER_GENRE:
                break
            if not args.dry_run and not args.skip_availability:
                if not is_available(item["ytid"]):
                    continue
            index += 1
            file_id = f"{index:04d}"
            tempo_bpm = item["tempo_bpm"]  # BPM anotado en caption (rarísimo)

            if not args.dry_run:
                raw_path = raw_dir / f"{file_id}.wav"
                tmp_path = raw_dir / f"{file_id}_norm.wav"
                if not download_clip(item, raw_path):
                    print(f"   ✗ {item['ytid']} no descargable, se omite")
                    index -= 1
                    continue
                if not normalize_to_32k_mono(raw_path, tmp_path):
                    index -= 1
                    continue
                if not verify_samples(tmp_path):
                    print(f"   ✗ {file_id} no tiene {EXPECTED_SAMPLES} muestras, se descarta")
                    tmp_path.unlink(missing_ok=True)
                    index -= 1
                    continue
                # Tempo estimado del audio si el caption no trae BPM numérico.
                if tempo_bpm is None and not args.no_bpm_estimate:
                    tempo_bpm = estimate_bpm(tmp_path)
                tempo_tok = str(tempo_bpm) if tempo_bpm else "na"
                fname = f"{file_id}_{genre}_{tempo_tok}.wav"
                final_path = out_dir / fname
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(final_path))
                print(f"   ✓ {fname}")
            else:
                tempo_tok = str(tempo_bpm) if tempo_bpm else "na"

            selected.append({
                "id": file_id,
                "ytid": item["ytid"],
                "caption": item["caption"],
                "genre": genre,
                "tempo_bpm": tempo_bpm if tempo_bpm else "",
                "instruments": "; ".join(item["instruments"]),
            })
            kept += 1
        bpm_count = sum(1 for s in selected if s["genre"] == genre and s["tempo_bpm"] != "")
        print(f"   {genre:<11} {kept}/{PER_GENRE} seleccionados ({bpm_count} con tempo)")

    print(f"\n9) Escribiendo metadata → {meta_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["id", "ytid", "caption", "genre", "tempo_bpm", "instruments"]
        )
        writer.writeheader()
        writer.writerows(selected)

    if not args.dry_run and raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"\nListo: {len(selected)} clips ({'simulado' if args.dry_run else 'descargados'}).")
    if args.dry_run:
        print("Ejecuta sin --dry-run para descargar el audio real.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="testset", help="Directorio de salida (default: testset/)")
    parser.add_argument("--csv", default=None, help="Ruta local a musiccaps-public.csv")
    parser.add_argument("--dry-run", action="store_true", help="Solo selección, sin descargar")
    parser.add_argument("--skip-availability", action="store_true",
                        help="No verificar disponibilidad con yt-dlp --simulate")
    parser.add_argument("--no-bpm-estimate", action="store_true",
                        help="No estimar el BPM con librosa tras la descarga")
    main_args = parser.parse_args()
    build(main_args)


if __name__ == "__main__":
    main()
