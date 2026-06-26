#!/usr/bin/env python3
"""
kad_score.py — Kernel Audio Distance (KAD)

Mide la discrepancia entre distribuciones de audio real y generado usando
Maximum Mean Discrepancy (MMD) con kernel RBF sobre los MISMOS embeddings que
usa FAD (vggish | clap | mel). A diferencia de FAD, MMD no asume que los
embeddings sean gaussianos, por lo que es más robusto a distribuciones no
normales y a tamaños de muestra pequeños.

Definición
----------
Para conjuntos X = {x_1..x_n} (real) e Y = {y_1..y_m} (generado), el estimador
INSESGADO de MMD² con kernel k es:

  MMD²(X,Y) = 1/(n(n-1)) Σ_{i≠j} k(x_i,x_j)
            + 1/(m(m-1)) Σ_{i≠j} k(y_i,y_j)
            - 2/(nm)     Σ_{i,j} k(x_i,y_j)

con kernel RBF:  k(a,b) = exp(-‖a-b‖² / (2σ²)).

El ancho de banda σ se fija con la HEURÍSTICA DE LA MEDIANA: σ = mediana de las
distancias euclidianas por pares dentro del conjunto real. Menor KAD = mayor
similitud entre audio real y generado.

Uso rápido:
    python kad_score.py --real audio/real --generated audio/gen --extractor mel

Uso como módulo:
    from kad_score import compute_kad
    results = compute_kad(Path("real/"), Path("gen/"), extractor_name="vggish")
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

# Reutilizamos extractores y utilidades de archivo de FAD (mismo espacio de
# embeddings, por diseño del protocolo). Funciona tanto importado como parte del
# paquete (hybrid_music_engine.new_metrics) como ejecutado de forma suelta.
try:  # pragma: no cover - depende del contexto de import
    from .fad_score import EXTRACTORS, list_audio_files
except ImportError:  # ejecución como script suelto (sys.path incluye este dir)
    from fad_score import EXTRACTORS, list_audio_files

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Matemáticas KAD ─────────────────────────────────────────────────────────

def median_bandwidth(reference: np.ndarray) -> float:
    """
    σ = mediana de las distancias euclidianas por pares del conjunto de
    referencia (heurística de la mediana). Si hay <2 puntos o la mediana es 0,
    cae a 1.0 para evitar un kernel degenerado.
    """
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        return 1.0
    from scipy.spatial.distance import pdist

    dists = pdist(reference, metric="euclidean")
    if dists.size == 0:
        return 1.0
    sigma = float(np.median(dists))
    return sigma if sigma > 1e-12 else 1.0


def _rbf_gram(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Matriz de Gram RBF k(a_i, b_j) = exp(-gamma·‖a_i-b_j‖²)."""
    # ‖a-b‖² = ‖a‖² + ‖b‖² − 2 a·b  (estable y vectorizado)
    a_sq = np.sum(a * a, axis=1)[:, None]
    b_sq = np.sum(b * b, axis=1)[None, :]
    sq_dists = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
    return np.exp(-gamma * sq_dists)


def mmd2_unbiased(real: np.ndarray, gen: np.ndarray, sigma: float) -> float:
    """
    Estimador insesgado de MMD² con kernel RBF y ancho de banda σ.

    Puede ser ligeramente negativo por la corrección de sesgo cuando las
    distribuciones son casi idénticas; eso es esperado y se reporta tal cual.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if real.ndim != 2 or gen.ndim != 2:
        raise ValueError("real y gen deben ser arrays 2D (n_muestras, dim).")
    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"Dimensiones incompatibles: real {real.shape[1]} vs gen {gen.shape[1]}."
        )
    n, m = real.shape[0], gen.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Se necesitan al menos 2 muestras por conjunto para MMD insesgado.")

    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = _rbf_gram(real, real, gamma)
    k_yy = _rbf_gram(gen, gen, gamma)
    k_xy = _rbf_gram(real, gen, gamma)

    # Términos intra-conjunto excluyendo la diagonal (i≠j).
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = k_xy.sum() / (n * m)

    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def compute_kad_from_embeddings(
    real_emb: np.ndarray,
    gen_emb: np.ndarray,
    *,
    sigma: float | None = None,
) -> dict:
    """
    Núcleo reutilizable: calcula KAD a partir de embeddings ya extraídos.
    `sigma` opcional; si es None se usa la heurística de la mediana sobre `real_emb`.
    """
    real_emb = np.asarray(real_emb, dtype=np.float64)
    gen_emb = np.asarray(gen_emb, dtype=np.float64)
    bw = float(sigma) if sigma is not None else median_bandwidth(real_emb)
    kad = mmd2_unbiased(real_emb, gen_emb, bw)
    return {
        "kad": kad,
        "sigma": bw,
        "real_n": int(real_emb.shape[0]),
        "generated_n": int(gen_emb.shape[0]),
        "embedding_dim": int(real_emb.shape[1]),
    }


# ── Extracción por clip (un embedding por archivo) ──────────────────────────

def _pool(emb: np.ndarray, mode: str) -> np.ndarray:
    """Reduce un embedding por-frame (T, D) a un vector (D,)."""
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim == 1:
        return emb
    if mode == "mean":
        return emb.mean(axis=0)
    if mode == "max":
        return emb.max(axis=0)
    raise ValueError(f"pool desconocido: {mode!r} (usa 'mean' o 'max').")


def extract_clip_embeddings(files: list[Path], extractor, pool: str = "mean") -> np.ndarray:
    """Un vector por archivo (pooling sobre frames). Salta los que fallen."""
    rows: list[np.ndarray] = []
    errors: list[str] = []
    for i, f in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), f.name)
        try:
            rows.append(_pool(extractor.extract(f), pool))
        except Exception as exc:  # noqa: BLE001
            log.warning("Error en %s: %s — omitido", f.name, exc)
            errors.append(f.name)
    if not rows:
        raise RuntimeError("No se pudo extraer ningún embedding para KAD.")
    if errors:
        log.warning("%d archivo(s) omitido(s) por errores.", len(errors))
    return np.vstack(rows)


# ── Pipeline principal ──────────────────────────────────────────────────────

def compute_kad(
    real_folder: Path,
    gen_folder: Path,
    extractor_name: str = "vggish",
    extractor_kwargs: dict | None = None,
    output_dir: Path = Path("metrics/kad"),
    pool: str = "mean",
) -> dict:
    """
    Pipeline completo de KAD: extrae embeddings por clip y calcula MMD²-RBF.

    Devuelve dict con kad, sigma, metadatos y rutas de archivos guardados.
    """
    extractor_kwargs = extractor_kwargs or {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Inicializando extractor: %s", extractor_name)
    extractor = EXTRACTORS[extractor_name](**extractor_kwargs)

    real_files = list_audio_files(real_folder)
    gen_files = list_audio_files(gen_folder)
    log.info("Real: %d archivos | Generado: %d archivos", len(real_files), len(gen_files))

    real_emb = extract_clip_embeddings(real_files, extractor, pool=pool)
    gen_emb = extract_clip_embeddings(gen_files, extractor, pool=pool)

    core = compute_kad_from_embeddings(real_emb, gen_emb)
    log.info("KAD (%s) = %.6f  (σ=%.4f)", extractor_name, core["kad"], core["sigma"])

    result_file = output_dir / f"kad_{extractor_name}_{timestamp}.json"
    results = {
        **core,
        "extractor": extractor_name,
        "pool": pool,
        "real_folder": str(real_folder.resolve()),
        "generated_folder": str(gen_folder.resolve()),
        "real_n_files": len(real_files),
        "generated_n_files": len(gen_files),
        "timestamp": timestamp,
        "saved_files": {"results_json": str(result_file)},
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Resultados JSON → %s", result_file)

    history_file = output_dir / "history.jsonl"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in results.items() if k != "saved_files"}) + "\n")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kernel Audio Distance (KAD) — MMD con kernel RBF sobre embeddings de audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python kad_score.py --real audio/real --generated audio/gen --extractor mel\n"
            "  python kad_score.py --real audio/real --generated audio/gen --extractor vggish\n"
        ),
    )
    p.add_argument("--real", type=Path, required=True, help="Carpeta con audio real/referencia.")
    p.add_argument("--generated", type=Path, required=True, help="Carpeta con audio generado.")
    p.add_argument("--extractor", choices=list(EXTRACTORS), default="vggish",
                   help="Extractor de embeddings (igual que FAD).")
    p.add_argument("--pool", choices=["mean", "max"], default="mean",
                   help="Pooling de frames a un vector por clip.")
    p.add_argument("--output-dir", type=Path, default=Path("metrics/kad"))
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps (solo vggish/clap).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    extractor_kwargs: dict = {}
    if args.extractor in ("vggish", "clap", "pann"):
        extractor_kwargs["device"] = args.device

    results = compute_kad(
        real_folder=args.real,
        gen_folder=args.generated,
        extractor_name=args.extractor,
        extractor_kwargs=extractor_kwargs,
        output_dir=args.output_dir,
        pool=args.pool,
    )

    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  KAD ({results['extractor']}) = {results['kad']:.6f}")
    print(f"  σ (mediana) = {results['sigma']:.4f}   dim = {results['embedding_dim']}")
    print(sep)
    print("  ↓ KAD menor = mayor similitud con audio real\n")


if __name__ == "__main__":
    main()
