#!/usr/bin/env python3
"""
clap_score.py — CLAP Score

Mide coherencia semántica entre audio generado y su prompt textual usando el
espacio vectorial compartido de CLAP (Contrastive Language-Audio Pretraining).
No requiere audio de referencia (reference-free).

Uso rápido (pares en CSV):
    python clap_score.py --pairs pares.csv

Uso rápido (carpeta + archivo de prompts):
    python clap_score.py --audio-folder audio/gen --prompts prompts.txt

Uso como módulo:
    from clap_score import compute_clap_score
    results = compute_clap_score(pairs=[("audio.wav", "jazz piano solo")], ...)

Formato CSV esperado (sin encabezado, o con encabezado audio,text):
    audio/gen/track_01.wav,jazz piano melancholic
    audio/gen/track_02.wav,upbeat electronic dance music

Formato TXT de prompts (un prompt por línea, mismo orden que archivos de audio):
    jazz piano melancholic
    upbeat electronic dance music
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


# ── Matemáticas CLAP Score ────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    similitud = (a · b) / (‖a‖ · ‖b‖)

    Retorna valor en [−1, 1]; para embeddings CLAP normalizados es [0, 1].
    """
    a = a.ravel()
    b = b.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Modelos CLAP ──────────────────────────────────────────────────────────────

class CLAPModel:
    """
    Wrapper sobre laion-clap que expone:
      - encode_audio(path) → np.ndarray (1, D)
      - encode_text(prompt) → np.ndarray (1, D)

    Variantes:
      "clap"       — modelo general (AudioSet + FreeSound)
      "clap-music" — modelo especializado en música (recomendado para MusicGen)

    Requiere: pip install laion-clap
    """

    CHECKPOINTS = {
        "clap":       None,                                        # descarga automática
        "clap-music": "music_audioset_epoch_15_esc_90.14.pt",     # checkpoint música
    }

    def __init__(self, variant: str = "clap", device: str = "cpu"):
        if variant not in self.CHECKPOINTS:
            raise ValueError(
                f"Variante desconocida: '{variant}'. Usa: {list(self.CHECKPOINTS)}"
            )
        try:
            import laion_clap
        except ImportError:
            _dep_error("laion_clap", "pip install laion-clap")

        import laion_clap

        self.variant = variant
        self.device = device
        log.info("Cargando modelo CLAP variante='%s' device='%s'", variant, device)

        ckpt = self.CHECKPOINTS[variant]
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        if ckpt:
            self.model.load_ckpt(ckpt)
        else:
            self.model.load_ckpt()

        log.info("Modelo CLAP listo.")

    def encode_audio(self, audio_path: Path) -> np.ndarray:
        """Embedding de audio normalizado, shape (1, D)."""
        emb = self.model.get_audio_embedding_from_filelist(
            x=[str(audio_path)], use_tensor=False
        )
        return _l2_normalize(emb)

    def encode_text(self, prompt: str) -> np.ndarray:
        """Embedding de texto normalizado, shape (1, D)."""
        emb = self.model.get_text_embedding([prompt], use_tensor=False)
        return _l2_normalize(emb)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normaliza cada fila a norma unitaria."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return x / norms


def _dep_error(package: str, install_cmd: str) -> None:
    print(f"\n[ERROR] Paquete requerido no encontrado: {package}")
    print(f"        Instalar con: {install_cmd}\n")
    sys.exit(1)


# ── Carga de pares audio-texto ────────────────────────────────────────────────

Pair = tuple[Path, str]


def load_pairs_from_csv(csv_path: Path) -> list[Pair]:
    """
    Lee pares (audio_path, prompt) desde un CSV.
    Acepta archivos con o sin fila de encabezado (audio, text).
    """
    pairs: list[Pair] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            audio_col, text_col = row[0].strip(), row[1].strip()
            if i == 0 and audio_col.lower() in ("audio", "audio_path", "file"):
                continue  # saltar encabezado
            pairs.append((Path(audio_col), text_col))
    if not pairs:
        raise ValueError(f"No se encontraron pares válidos en: {csv_path}")
    return pairs


def load_pairs_from_json(json_path: Path) -> list[Pair]:
    """
    Lee pares desde JSON. Formato esperado:
        [{"audio": "ruta.wav", "text": "prompt"}, ...]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data:
        audio = item.get("audio") or item.get("audio_path") or item.get("file")
        text = item.get("text") or item.get("prompt") or item.get("caption")
        if audio and text:
            pairs.append((Path(audio), str(text)))
    if not pairs:
        raise ValueError(f"No se encontraron pares válidos en: {json_path}")
    return pairs


def load_pairs_from_folder(audio_folder: Path, prompts_file: Path) -> list[Pair]:
    """
    Construye pares emparejando archivos de audio (orden alfabético) con
    las líneas del archivo de prompts (una por línea).
    """
    audio_files = sorted(
        p for p in audio_folder.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(f"No hay audio en: {audio_folder}")

    prompts = [
        line.rstrip("\n")
        for line in prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(audio_files) != len(prompts):
        raise ValueError(
            f"Cantidad de audios ({len(audio_files)}) ≠ cantidad de prompts "
            f"({len(prompts)}). Deben coincidir línea a línea."
        )

    return list(zip(audio_files, prompts))


# ── Pipeline principal ────────────────────────────────────────────────────────

def compute_clap_score(
    pairs: list[Pair],
    model_variant: str = "clap",
    device: str = "cpu",
    output_dir: Path = Path("metrics/clap"),
    use_cache: bool = True,
    run_label: str = "",
    accept_threshold: float = 0.25,
) -> dict:
    """
    Pipeline completo de CLAP Score.

    Parámetros
    ----------
    pairs            : lista de (audio_path, prompt_texto)
    model_variant    : "clap" | "clap-music"
    device           : "cpu" | "cuda" | "mps"
    output_dir       : carpeta raíz para resultados y caché de embeddings
    use_cache        : si True, reutiliza embeddings de audio ya calculados
    run_label        : etiqueta opcional para identificar la corrida (ej. "modelo_v2")
    accept_threshold : umbral de 'alineación aceptable'; se reporta el % de clips
                       con CLAP-score por encima de este valor (protocolo: 0.25)

    Retorna
    -------
    dict con clap_score, métricas por par, y rutas de archivos guardados
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "embeddings_cache" / model_variant

    model = CLAPModel(variant=model_variant, device=device)

    per_pair: list[dict] = []
    similarities: list[float] = []
    errors: list[str] = []

    for i, (audio_path, prompt) in enumerate(pairs, 1):
        audio_path = Path(audio_path)
        log.info("[%d/%d] %s", i, len(pairs), audio_path.name)

        try:
            # ── Embedding de audio (con caché opcional) ───────────────────────
            cache_path = (
                cache_dir / f"{audio_path.stem}.npy"
                if use_cache else None
            )

            if cache_path is not None and cache_path.exists():
                emb_audio = np.load(cache_path)
            else:
                emb_audio = model.encode_audio(audio_path)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_path, emb_audio)

            # ── Embedding de texto ────────────────────────────────────────────
            emb_text = model.encode_text(prompt)

            # ── Similitud coseno ──────────────────────────────────────────────
            sim = cosine_similarity(emb_audio, emb_text)
            similarities.append(sim)

            per_pair.append({
                "audio": str(audio_path),
                "prompt": prompt,
                "similarity": sim,
            })
            log.info("  similitud coseno = %.4f", sim)

        except Exception as exc:
            log.warning("  Error procesando %s: %s — omitido", audio_path.name, exc)
            errors.append(str(audio_path))

    if not similarities:
        raise RuntimeError("No se pudo calcular ninguna similitud. Revisa los archivos.")

    sims = np.asarray(similarities, dtype=np.float64)
    clap_score = float(sims.mean())
    clap_std = float(sims.std())
    clap_min = float(sims.min())
    clap_max = float(sims.max())
    # Indicador de 'alineación aceptable': % de clips por encima del umbral.
    n_above = int(np.count_nonzero(sims > accept_threshold))
    pct_above = float(100.0 * n_above / sims.size)

    log.info("CLAP Score (%s) = %.4f  (σ=%.4f, min=%.4f, max=%.4f, %%>%.2f=%.1f%%)",
             model_variant, clap_score, clap_std, clap_min, clap_max,
             accept_threshold, pct_above)

    # ── Guardar resultados ────────────────────────────────────────────────────

    tag = f"_{run_label}" if run_label else ""
    result_file = output_dir / f"clap_{model_variant}{tag}_{timestamp}.json"

    results = {
        "clap_score": clap_score,
        "clap_std": clap_std,
        "clap_min": clap_min,
        "clap_max": clap_max,
        "accept_threshold": float(accept_threshold),
        "n_above_threshold": n_above,
        "pct_above_threshold": pct_above,
        "model_variant": model_variant,
        "n_pairs": len(similarities),
        "n_errors": len(errors),
        "run_label": run_label,
        "timestamp": timestamp,
        "per_pair": per_pair,
        "saved_files": {},
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Resultados JSON → %s", result_file)
    results["saved_files"]["results_json"] = str(result_file)

    # CSV con puntuaciones por par (fácil de abrir en Excel / pandas)
    csv_file = output_dir / f"per_pair_{model_variant}{tag}_{timestamp}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "prompt", "similarity"])
        writer.writeheader()
        writer.writerows(per_pair)
    log.info("Por par CSV → %s", csv_file)
    results["saved_files"]["per_pair_csv"] = str(csv_file)

    # Historial acumulado (append-safe, una línea por corrida)
    history_file = output_dir / "history.jsonl"
    history_entry = {k: v for k, v in results.items()
                     if k not in ("per_pair", "saved_files")}
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")
    log.info("Historial → %s", history_file)

    return results


def compare_multiple(
    pairs_per_run: dict[str, list[Pair]],
    **kwargs,
) -> list[dict]:
    """
    Evalúa múltiples conjuntos de pares (ej. distintos checkpoints).
    pairs_per_run: {etiqueta: [(audio_path, prompt), ...]}
    """
    all_results = []
    for label, pairs in pairs_per_run.items():
        log.info("═" * 50)
        log.info("Evaluando: %s (%d pares)", label, len(pairs))
        r = compute_clap_score(pairs, run_label=label, **kwargs)
        all_results.append(r)

    all_results.sort(key=lambda x: x["clap_score"], reverse=True)
    log.info("═" * 50)
    log.info("Ranking (↑ CLAP Score = mejor alineación texto-audio):")
    for i, r in enumerate(all_results, 1):
        log.info("  %d. CLAP=%.4f  %s", i, r["clap_score"], r["run_label"])

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLAP Score — coherencia semántica audio↔texto (reference-free)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python clap_score.py --pairs pares.csv\n"
            "  python clap_score.py --audio-folder audio/gen --prompts prompts.txt\n"
            "  python clap_score.py --pairs pares.json --model clap-music --device cuda\n"
        ),
    )

    # Fuente de pares (mutuamente excluyentes)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pairs", type=Path, metavar="FILE",
        help="CSV o JSON con pares (audio_path, prompt)",
    )
    src.add_argument(
        "--audio-folder", type=Path, metavar="DIR",
        help="Carpeta con audios generados (usar junto con --prompts)",
    )

    p.add_argument(
        "--prompts", type=Path, metavar="FILE",
        help="Archivo .txt con un prompt por línea (requerido si se usa --audio-folder)",
    )
    p.add_argument(
        "--model", choices=["clap", "clap-music"], default="clap",
        dest="model_variant",
        help="Variante CLAP: general o especializado en música",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("metrics/clap"),
        help="Directorio raíz para resultados y caché",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Dispositivo torch: cpu | cuda | mps",
    )
    p.add_argument(
        "--label", default="", dest="run_label",
        help="Etiqueta para identificar la corrida en el historial",
    )
    p.add_argument(
        "--threshold", type=float, default=0.25, dest="accept_threshold",
        help="Umbral de 'alineación aceptable'; se reporta el %% de clips por encima",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Recalcular embeddings de audio aunque exista caché",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Resolver fuente de pares
    if args.pairs is not None:
        path = args.pairs
        if path.suffix.lower() == ".json":
            pairs = load_pairs_from_json(path)
        else:
            pairs = load_pairs_from_csv(path)
    else:
        if args.prompts is None:
            print("[ERROR] --audio-folder requiere también --prompts")
            sys.exit(1)
        pairs = load_pairs_from_folder(args.audio_folder, args.prompts)

    log.info("Pares cargados: %d", len(pairs))

    results = compute_clap_score(
        pairs=pairs,
        model_variant=args.model_variant,
        device=args.device,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        run_label=args.run_label,
        accept_threshold=args.accept_threshold,
    )

    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  CLAP Score ({results['model_variant']}) = {results['clap_score']:.4f}")
    print(f"  σ={results['clap_std']:.4f}  "
          f"min={results['clap_min']:.4f}  max={results['clap_max']:.4f}")
    print(f"  %>{results['accept_threshold']:.2f} (alineación aceptable) = "
          f"{results['pct_above_threshold']:.1f}%")
    print(f"  Pares evaluados: {results['n_pairs']}  "
          f"Errores: {results['n_errors']}")
    print(sep)
    print("  ↑ CLAP Score mayor = mejor alineación audio↔texto\n")


if __name__ == "__main__":
    main()
