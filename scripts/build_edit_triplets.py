#!/usr/bin/env python3
"""build_edit_triplets.py — Generador del banco de ternas de EDICIÓN.

Construye ternas ``{fuente, instrucción, objetivo}`` con objetivos derivados de
forma **controlada y transparente** a partir de pistas fuente, para alimentar el
benchmark de edición (``hybrid_music_engine/new_metrics/edit_benchmark.py``).

Por qué la transparencia importa
--------------------------------
Cómo se construye el audio objetivo cambia las conclusiones, sesgos y dificultad
de la tarea. Aquí cada objetivo se genera por un método explícito (registrado en
la metadata): operaciones de DSP deterministas o manipulación de stems con
Demucs. NO son "pares reales" anotados por humanos; son objetivos sintéticos
controlados (sesgo documentado en la sección de datos).

Modalidad / estandarización
---------------------------
Las fuentes se normalizan a un formato canónico (32 kHz mono) antes de editar.
Las métricas de edición (FAD/CLAP/preservación/mezcla) viven en el dominio de
audio; el MIDI no puede representar timbre ni mezcla, por lo que el corpus se
estandariza a un registro que CONSERVA ambas modalidades cuando aplica (WAV +
MIDI + tokens), eligiendo la representación según la operación. Ver
``results/edit_data_section.tex``.

Operaciones
-----------
DSP (siempre disponibles):
  slow_down, speed_up : time-stretch (preserva tono); objetivo = tempo distinto.
  quieter             : ganancia -6 dB; objetivo = mezcla más baja.
  fade_out            : fundido de salida; objetivo = cola atenuada.
Stems (requieren ``demucs``):
  remove_drums, remove_vocals : objetivo = mezcla sin esa capa.
  isolate_bass                : objetivo = solo el bajo.

Uso:
  python build_edit_triplets.py --sources testset --out edit_triplets --dsp-only --limit 10
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

TARGET_SR = 32000
TARGET_LEN = TARGET_SR * 10  # 10 s canónicos


# ── Utilidades de audio (numpy mono) ─────────────────────────────────────────

def rms(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(y * y))) if y.size else 0.0


def tail_energy(y: np.ndarray, sr: int, seconds: float = 2.0) -> float:
    n = int(seconds * sr)
    return rms(y[-n:]) if y.size >= n else rms(y)


def fit_length(y: np.ndarray, n: int = TARGET_LEN) -> np.ndarray:
    if y.shape[-1] < n:
        return np.pad(y, (0, n - y.shape[-1]))
    return y[..., :n]


def stem_op_vacuous(source: np.ndarray, target: np.ndarray,
                    min_change: float = 0.05, min_abs: float = 1e-4) -> bool:
    """Criterio de exclusión para operaciones de stems.

    Una operación es vacua si (a) el objetivo queda casi en silencio (p. ej.
    aislar una capa ausente en la ventana) o (b) el cambio relativo de energía
    respecto a la fuente es despreciable (p. ej. quitar batería en un intro sin
    batería). Evita ternas donde la operación no aplica realmente.
    """
    s, t = rms(source), rms(target)
    if t < min_abs or s < min_abs:
        return True
    return abs(t - s) / s < min_change


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    import librosa
    tempo = librosa.feature.tempo(y=np.asarray(y, dtype=np.float32), sr=sr)
    return float(np.asarray(tempo).reshape(-1)[0])


# ── Operaciones de DSP (objetivo determinista) ───────────────────────────────

def op_gain(y: np.ndarray, db: float) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * (10.0 ** (db / 20.0))


def op_fade_out(y: np.ndarray, sr: int, seconds: float = 3.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    n = min(int(seconds * sr), y.shape[-1])
    if n > 0:
        ramp = np.linspace(1.0, 0.0, n)
        y[..., -n:] *= ramp
    return y


def op_time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate < 1 ralentiza (objetivo más lento); > 1 acelera. Recorta/rellena a 10 s."""
    import librosa
    stretched = librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=rate)
    return fit_length(stretched.astype(np.float64))


# ── Registro de operaciones ──────────────────────────────────────────────────
# kind: "dsp" (determinista) | "stem" (requiere demucs)
# attr: medida del atributo objetivo para "éxito de operación"
# direction: hacia dónde debe moverse el atributo (decrease/increase)
OPERATIONS: dict[str, dict[str, Any]] = {
    "slow_down": {
        "kind": "dsp", "attr": "bpm", "direction": "decrease", "edited_stems": [],
        "instructions": ["haz el tempo más lento", "ralentiza la pista", "baja la velocidad"],
        "build": lambda y, sr: op_time_stretch(y, sr, rate=0.82),
    },
    "speed_up": {
        "kind": "dsp", "attr": "bpm", "direction": "increase", "edited_stems": [],
        "instructions": ["haz el tempo más rápido", "acelera la pista", "sube la velocidad"],
        "build": lambda y, sr: op_time_stretch(y, sr, rate=1.20),
    },
    "quieter": {
        "kind": "dsp", "attr": "rms", "direction": "decrease", "edited_stems": [],
        "instructions": ["baja el volumen general", "suaviza la mezcla", "hazlo más bajo"],
        "build": lambda y, sr: op_gain(y, db=-6.0),
    },
    "fade_out": {
        "kind": "dsp", "attr": "tail_energy", "direction": "decrease", "edited_stems": [],
        "instructions": ["añade un fundido de salida", "termina con un fade out"],
        "build": lambda y, sr: op_fade_out(y, sr, seconds=3.0),
    },
    "remove_drums": {
        "kind": "stem", "attr": "rms", "direction": "decrease", "edited_stems": ["drums"],
        "instructions": ["quita la batería", "elimina la percusión"],
        "stem_keep": ["bass", "vocals", "other"],
    },
    "remove_vocals": {
        "kind": "stem", "attr": "rms", "direction": "decrease", "edited_stems": ["vocals"],
        "instructions": ["quita la voz", "elimina las voces"],
        "stem_keep": ["drums", "bass", "other"],
    },
    "isolate_bass": {
        # Aislar el bajo quita drums+vocals+other -> el RMS total BAJA.
        "kind": "stem", "attr": "rms", "direction": "decrease", "edited_stems": ["drums", "vocals", "other"],
        "instructions": ["deja solo el bajo", "aísla la línea de bajo"],
        "stem_keep": ["bass"],
    },
}

ATTR_FNS: dict[str, Callable[[np.ndarray, int], float]] = {
    "rms": lambda y, sr: rms(y),
    "tail_energy": lambda y, sr: tail_energy(y, sr),
    "bpm": estimate_bpm,
}


def instruction_for(op_name: str, rng: random.Random) -> str:
    return rng.choice(OPERATIONS[op_name]["instructions"])


# ── Construcción de objetivos por stems (Demucs, opcional) ───────────────────

def build_stem_target(source_path: Path, op_name: str, tmp_dir: Path) -> np.ndarray:
    """Separa con Demucs y suma los stems a conservar. Requiere `demucs`."""
    import soundfile as sf
    from hybrid_music_engine.audio.stems import separate_stems

    keep = OPERATIONS[op_name]["stem_keep"]
    result = separate_stems(source_path, tmp_dir, device="cpu")
    stems = result.get("stems", result)  # dict nombre -> ruta
    mix = None
    for name in keep:
        path = stems.get(name) if isinstance(stems, dict) else None
        if not path:
            continue
        y, _sr = sf.read(str(path))
        y = y.mean(axis=1) if y.ndim > 1 else y
        mix = y if mix is None else mix[: len(y)] + y[: len(mix)]
    if mix is None:
        raise RuntimeError(f"Demucs no devolvió stems esperados para {op_name}")
    return fit_length(np.asarray(mix, dtype=np.float64))


# ── Orquestación ─────────────────────────────────────────────────────────────

def load_source(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return fit_length(np.asarray(y, dtype=np.float64)), sr


def _genre_from_name(stem: str) -> str:
    """Testset usa nombres NNNN_genero_BPM; extrae el género si está."""
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def build_triplets_moisesdb(
    tracks: list[Path],
    ops: list[str],
    out_dir: Path,
    *,
    seed: int = 42,
    no_bpm: bool = False,
    segment: str = "energy",
) -> list[dict[str, Any]]:
    """Construye ternas usando los stems REALES de MoisesDB (sin Demucs).

    Para operaciones de stems el objetivo se arma sumando las capas reales a
    conservar; para DSP se aplica a la mezcla. La fuente es la mezcla escrita a
    disco. La columna ``stems_origin`` queda en ``real``.
    """
    import soundfile as sf
    import moisesdb_adapter as moises

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    (out_dir / "targets").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tid = 0

    for track in tracks:
        rec = moises.load_track_stems(track, segment=segment)
        mixture, stems = rec["mixture"], rec["stems"]
        genre = moises.track_genre(track)
        src_path = out_dir / "sources" / f"{track.name}.wav"
        sf.write(str(src_path), fit_length(mixture).astype(np.float32), TARGET_SR)

        for op_name in ops:
            op = OPERATIONS[op_name]
            if op["kind"] == "dsp":
                target = op["build"](mixture, TARGET_SR)
            else:
                keep = op.get("stem_keep", [])
                present = [k for k in keep if k in stems]
                if not present:
                    print(f"  [omitido] {track.name} · {op_name}: sin stems a conservar")
                    continue
                target = fit_length(sum(stems[k] for k in present))

            # Exclusión: operaciones de stems que no producen un cambio real
            # (intro sin la capa, capa ausente -> objetivo en silencio, etc.).
            if op["kind"] == "stem" and stem_op_vacuous(mixture, target):
                print(f"  [excluida] {track.name} · {op_name}: operación vacua (capa despreciable en la ventana)")
                continue

            tid += 1
            tgt_path = out_dir / "targets" / f"{tid:04d}_{op_name}.wav"
            sf.write(str(tgt_path), fit_length(target).astype(np.float32), TARGET_SR)

            attr = op["attr"]
            if attr == "bpm" and no_bpm:
                src_a = tgt_a = ""
            else:
                fn = ATTR_FNS[attr]
                src_a = round(fn(mixture, TARGET_SR), 4)
                tgt_a = round(fn(target, TARGET_SR), 4)

            rows.append({
                "id": f"{tid:04d}",
                "source_path": str(src_path),
                "target_path": str(tgt_path),
                "instruction": instruction_for(op_name, rng),
                "operation": op_name,
                "kind": op["kind"],
                "edited_stems": ";".join(op["edited_stems"]),
                "attr_name": attr,
                "attr_direction": op["direction"],
                "source_attr": src_a,
                "target_attr": tgt_a,
                "stems_origin": "real" if op["kind"] == "stem" else "n/a",
                "genre": genre,
                "target_method": ("dsp:" + op_name) if op["kind"] == "dsp"
                                 else ("moisesdb_real_stems:keep=" + "+".join(op.get("stem_keep", []))),
            })
            print(f"  ✓ {track.name} · {op_name} -> {tgt_path.name}")

    if rows:
        manifest = out_dir / "edit_triplets.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nManifest: {manifest}  ({len(rows)} ternas · stems reales)")
    return rows


def build_triplets(
    sources: list[Path],
    ops: list[str],
    out_dir: Path,
    *,
    seed: int = 42,
    no_bpm: bool = False,
) -> list[dict[str, Any]]:
    import soundfile as sf

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    (out_dir / "targets").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tid = 0

    for src in sources:
        y, _sr = load_source(src)
        for op_name in ops:
            op = OPERATIONS[op_name]
            try:
                if op["kind"] == "dsp":
                    target = op["build"](y, TARGET_SR)
                else:
                    target = build_stem_target(src, op_name, out_dir / "_demucs_tmp")
            except Exception as exc:  # noqa: BLE001
                print(f"  [omitido] {src.name} · {op_name}: {exc}")
                continue

            tid += 1
            tgt_path = out_dir / "targets" / f"{tid:04d}_{op_name}.wav"
            sf.write(str(tgt_path), fit_length(target).astype(np.float32), TARGET_SR)

            attr = op["attr"]
            if attr == "bpm" and no_bpm:
                src_a = tgt_a = ""
            else:
                fn = ATTR_FNS[attr]
                src_a = round(fn(y, TARGET_SR), 4)
                tgt_a = round(fn(target, TARGET_SR), 4)

            rows.append({
                "id": f"{tid:04d}",
                "source_path": str(src),
                "target_path": str(tgt_path),
                "instruction": instruction_for(op_name, rng),
                "operation": op_name,
                "kind": op["kind"],
                "edited_stems": ";".join(op["edited_stems"]),
                "attr_name": attr,
                "attr_direction": op["direction"],
                "source_attr": src_a,
                "target_attr": tgt_a,
                "stems_origin": "estimated" if op["kind"] == "stem" else "n/a",
                "genre": _genre_from_name(src.stem),
                "target_method": ("dsp:" + op_name) if op["kind"] == "dsp" else ("demucs_stems:keep=" + "+".join(op.get("stem_keep", []))),
            })
            print(f"  ✓ {src.name} · {op_name} -> {tgt_path.name}")

    manifest = out_dir / "edit_triplets.csv"
    if rows:
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nManifest: {manifest}  ({len(rows)} ternas)")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera el banco de ternas de edición.")
    parser.add_argument("--sources", type=Path, default=Path("testset"), help="Carpeta con WAV fuente.")
    parser.add_argument("--moisesdb", type=Path, default=None,
                        help="Raíz de MoisesDB: usa STEMS REALES en vez de Demucs.")
    parser.add_argument("--out", type=Path, default=Path("edit_triplets"), help="Carpeta de salida.")
    parser.add_argument("--limit", type=int, default=10, help="Máximo de fuentes/pistas a usar.")
    parser.add_argument("--ops", nargs="+", default=None, help="Operaciones (por defecto todas las aplicables).")
    parser.add_argument("--dsp-only", action="store_true", help="Solo operaciones de DSP (sin stems).")
    parser.add_argument("--no-bpm", action="store_true", help="No estimar BPM (más rápido).")
    parser.add_argument("--segment", choices=["start", "middle", "energy"], default="energy",
                        help="Ventana de 10 s a extraer de cada pista MoisesDB (energy recupera intros dispersos).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ops = args.ops or list(OPERATIONS.keys())
    if args.dsp_only:
        ops = [o for o in ops if OPERATIONS[o]["kind"] == "dsp"]

    if args.moisesdb:
        import moisesdb_adapter as moises
        tracks = moises.list_tracks(args.moisesdb)[: args.limit]
        if not tracks:
            print(f"No hay pistas de MoisesDB en {args.moisesdb}")
            return 1
        print(f"MoisesDB (stems reales): {len(tracks)} pistas · Operaciones: {ops} · segmento: {args.segment}")
        rows = build_triplets_moisesdb(tracks, ops, args.out, seed=args.seed,
                                       no_bpm=args.no_bpm, segment=args.segment)
    else:
        sources = sorted(Path(args.sources).glob("*.wav"))[: args.limit]
        if not sources:
            print(f"No hay WAV en {args.sources}")
            return 1
        print(f"Fuentes: {len(sources)} · Operaciones: {ops}")
        rows = build_triplets(sources, ops, args.out, seed=args.seed, no_bpm=args.no_bpm)

    print(f"\nListo: {len(rows)} ternas en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
