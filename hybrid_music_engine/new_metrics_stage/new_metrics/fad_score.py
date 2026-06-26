#!/usr/bin/env python3
"""
fad_score.py — Fréchet Audio Distance (FAD)

Mide similitud estadística entre distribuciones de audio real y generado
comparando sus embeddings con la distancia de Fréchet.

Uso rápido:
    python fad_score.py --real audio_raw/real --generated audio_raw/generated

Uso como módulo:
    from fad_score import compute_fad
    results = compute_fad(Path("real/"), Path("gen/"), extractor_name="vggish")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import linalg

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


# ── Matemáticas FAD ───────────────────────────────────────────────────────────

def frechet_distance(
    mu_r: np.ndarray,
    sigma_r: np.ndarray,
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    FAD = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2·√(Σ_r·Σ_g))

    eps: pequeña perturbación diagonal para estabilidad numérica de sqrtm.
    """
    diff = mu_r - mu_g
    term1 = float(diff @ diff)

    product = sigma_r @ sigma_g
    product += np.eye(product.shape[0]) * eps

    sqrt_prod, _ = linalg.sqrtm(product, disp=False)

    if np.iscomplexobj(sqrt_prod):
        imag_max = np.abs(sqrt_prod.imag).max()
        if imag_max > 1e-3:
            log.warning("sqrtm: parte imaginaria significativa (%.2e) — tomando Re(·)", imag_max)
        sqrt_prod = sqrt_prod.real

    term2 = float(np.trace(sigma_r + sigma_g - 2.0 * sqrt_prod))
    return term1 + term2


def compute_statistics(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (media, covarianza) para embeddings de shape (N, D)."""
    if embeddings.ndim != 2:
        raise ValueError(f"Se esperaba array 2D, se obtuvo shape {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 embeddings para estimar covarianza.")
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


# ── Extractores de embeddings ─────────────────────────────────────────────────

class VGGishExtractor:
    """
    VGGish — red entrenada en AudioSet (Google), embeddings de 128 dimensiones.
    Ventana: ~0.96 s. Requiere: pip install torchvggish
    """

    name = "vggish"
    dim = 128

    def __init__(self, device: str = "cpu"):
        try:
            import torch
            import torchvggish
        except ImportError:
            _dep_error("torchvggish", "pip install torchvggish")
        import torch
        import torchvggish

        self.torch = torch
        self.vggish_input = torchvggish.vggish_input
        self.model = torchvggish.vggish()
        self.model.eval()
        self.model.to(device)
        self.device = device

    def extract(self, audio_path: Path) -> np.ndarray:
        examples = self.vggish_input.wavfile_to_examples(str(audio_path))
        tensor = self.torch.tensor(examples, dtype=self.torch.float32).to(self.device)
        with self.torch.no_grad():
            emb = self.model(tensor)
        return emb.cpu().numpy()  # (n_frames, 128)


class CLAPExtractor:
    """
    CLAP — Contrastive Language-Audio Pretraining (LAION), embeddings de 512 dim.
    Requiere: pip install laion-clap
    """

    name = "clap"
    dim = 512

    def __init__(self, device: str = "cpu"):
        try:
            import laion_clap
        except ImportError:
            _dep_error("laion_clap", "pip install laion-clap")
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        self.model.load_ckpt()

    def extract(self, audio_path: Path) -> np.ndarray:
        emb = self.model.get_audio_embedding_from_filelist(
            x=[str(audio_path)], use_tensor=False
        )
        return emb  # (1, 512)


class MelExtractor:
    """
    Fallback ligero: vectores log-mel por frame (sin modelo preentrenado).
    Útil para pruebas rápidas. Requiere: pip install librosa
    """

    name = "mel"

    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        **_,
    ):
        try:
            import librosa  # noqa: F401
        except ImportError:
            _dep_error("librosa", "pip install librosa")
        import librosa

        self.librosa = librosa
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim = n_mels

    def extract(self, audio_path: Path) -> np.ndarray:
        y, _ = self.librosa.load(str(audio_path), sr=self.sr, mono=True)
        mel = self.librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
        )
        return self.librosa.power_to_db(mel).T  # (T, n_mels)


EXTRACTORS: dict[str, type] = {
    "vggish": VGGishExtractor,
    "clap": CLAPExtractor,
    "mel": MelExtractor,
}


def _dep_error(package: str, install_cmd: str) -> None:
    print(f"\n[ERROR] Paquete requerido no encontrado: {package}")
    print(f"        Instalar con: {install_cmd}\n")
    sys.exit(1)


# ── Pipeline principal ────────────────────────────────────────────────────────

def list_audio_files(folder: Path) -> list[Path]:
    files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos de audio en: {folder}\n"
            f"Extensiones soportadas: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )
    return files


def extract_embeddings(
    files: list[Path],
    extractor,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Extrae (o carga desde caché) embeddings para una lista de archivos."""
    if cache_path is not None and cache_path.exists():
        log.info("Cargando embeddings desde caché: %s", cache_path)
        return np.load(cache_path)

    all_embs: list[np.ndarray] = []
    errors: list[str] = []

    for i, f in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), f.name)
        try:
            emb = extractor.extract(f)
            if emb.ndim == 1:
                emb = emb[np.newaxis, :]
            all_embs.append(emb)
        except Exception as exc:
            log.warning("Error en %s: %s — omitido", f.name, exc)
            errors.append(f.name)

    if not all_embs:
        raise RuntimeError("No se pudo extraer ningún embedding. Revisa los archivos de audio.")
    if errors:
        log.warning("%d archivo(s) omitido(s) por errores.", len(errors))

    embeddings = np.vstack(all_embs)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        log.info("Embeddings guardados en caché → %s", cache_path)

    return embeddings


def compute_fad(
    real_folder: Path,
    gen_folder: Path,
    extractor_name: str = "vggish",
    extractor_kwargs: dict | None = None,
    output_dir: Path = Path("metrics/fad"),
    use_cache: bool = True,
) -> dict:
    """
    Pipeline completo de FAD.

    Parámetros
    ----------
    real_folder      : carpeta con audio de referencia (real)
    gen_folder       : carpeta con audio generado a evaluar
    extractor_name   : "vggish" | "clap" | "mel"
    extractor_kwargs : argumentos extra para el extractor (ej. device="cuda")
    output_dir       : carpeta raíz para resultados y caché
    use_cache        : si True, reutiliza embeddings ya calculados

    Retorna
    -------
    dict con fad, metadatos y rutas de archivos guardados
    """
    extractor_kwargs = extractor_kwargs or {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "embeddings_cache" / extractor_name

    # Construir extractor
    log.info("Inicializando extractor: %s", extractor_name)
    extractor_cls = EXTRACTORS[extractor_name]
    extractor = extractor_cls(**extractor_kwargs)

    # Embeddings — conjunto real
    real_files = list_audio_files(real_folder)
    log.info("Conjunto real   : %d archivos en %s", len(real_files), real_folder)
    real_cache = (cache_dir / f"real__{real_folder.name}.npy") if use_cache else None
    emb_r = extract_embeddings(real_files, extractor, cache_path=real_cache)

    # Embeddings — conjunto generado
    gen_files = list_audio_files(gen_folder)
    log.info("Conjunto generado: %d archivos en %s", len(gen_files), gen_folder)
    gen_cache = (cache_dir / f"gen__{gen_folder.name}.npy") if use_cache else None
    emb_g = extract_embeddings(gen_files, extractor, cache_path=gen_cache)

    # Parámetros gaussianos
    log.info("Calculando estadísticas de distribución…")
    mu_r, sigma_r = compute_statistics(emb_r)
    mu_g, sigma_g = compute_statistics(emb_g)

    # Distancia de Fréchet
    fad = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    log.info("FAD (%s) = %.6f", extractor_name, fad)

    # ── Guardar resultados ────────────────────────────────────────────────────

    # 1) JSON de resultados
    result_file = output_dir / f"fad_{extractor_name}_{timestamp}.json"
    results = {
        "fad": fad,
        "extractor": extractor_name,
        "real_folder": str(real_folder.resolve()),
        "generated_folder": str(gen_folder.resolve()),
        "real_n_files": len(real_files),
        "generated_n_files": len(gen_files),
        "embedding_dim": int(emb_r.shape[1]),
        "real_n_frames": int(emb_r.shape[0]),
        "generated_n_frames": int(emb_g.shape[0]),
        "timestamp": timestamp,
        "saved_files": {
            "results_json": str(result_file),
        },
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Resultados JSON → %s", result_file)

    # 2) Estadísticas numpy (mu, sigma de ambas distribuciones)
    stats_file = output_dir / f"stats_{extractor_name}_{timestamp}.npz"
    np.savez_compressed(
        stats_file,
        mu_r=mu_r, sigma_r=sigma_r,
        mu_g=mu_g, sigma_g=sigma_g,
    )
    log.info("Estadísticas .npz → %s", stats_file)
    results["saved_files"]["stats_npz"] = str(stats_file)

    # 3) Actualizar JSON con ruta de stats
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 4) Historial acumulado (append-safe)
    history_file = output_dir / "history.jsonl"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in results.items() if k != "saved_files"}) + "\n")
    log.info("Historial → %s", history_file)

    return results


# ── Comparación múltiple ──────────────────────────────────────────────────────

def compare_multiple(
    real_folder: Path,
    gen_folders: list[Path],
    **kwargs,
) -> list[dict]:
    """
    Compara un conjunto real contra múltiples carpetas de audio generado.
    Útil para comparar distintos checkpoints o configuraciones de modelo.
    """
    all_results = []
    for gen_folder in gen_folders:
        log.info("═" * 50)
        log.info("Evaluando: %s", gen_folder.name)
        r = compute_fad(real_folder, gen_folder, **kwargs)
        all_results.append(r)

    # Ranking
    all_results.sort(key=lambda x: x["fad"])
    log.info("═" * 50)
    log.info("Ranking (↓ FAD = mejor calidad generativa):")
    for i, r in enumerate(all_results, 1):
        log.info("  %d. FAD=%.4f  %s", i, r["fad"], Path(r["generated_folder"]).name)

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fréchet Audio Distance (FAD) — métrica de calidad de audio generativo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python fad_score.py --real audio_raw/real --generated audio_raw/gen\n"
            "  python fad_score.py --real audio_raw/real --generated audio_raw/gen "
            "--extractor mel --no-cache\n"
        ),
    )
    p.add_argument(
        "--real", type=Path, required=True,
        help="Carpeta con audio de referencia (real / ground truth)",
    )
    p.add_argument(
        "--generated", type=Path, nargs="+", required=True,
        help="Carpeta(s) con audio generado a evaluar (acepta múltiples)",
    )
    p.add_argument(
        "--extractor", choices=list(EXTRACTORS), default="vggish",
        help="Modelo de embeddings a usar",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("metrics/fad"),
        help="Directorio raíz para resultados y caché",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Dispositivo torch: cpu | cuda | mps  (solo vggish/clap)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Recalcular embeddings aunque exista caché",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    extractor_kwargs: dict = {}
    if args.extractor in ("vggish", "clap"):
        extractor_kwargs["device"] = args.device

    gen_folders: list[Path] = args.generated

    if len(gen_folders) == 1:
        results = compute_fad(
            real_folder=args.real,
            gen_folder=gen_folders[0],
            extractor_name=args.extractor,
            extractor_kwargs=extractor_kwargs,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
        )
        all_results = [results]
    else:
        all_results = compare_multiple(
            real_folder=args.real,
            gen_folders=gen_folders,
            extractor_name=args.extractor,
            extractor_kwargs=extractor_kwargs,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
        )

    # Resumen en consola
    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  {'Carpeta generada':<28} {'FAD':>10}")
    print(sep)
    for r in sorted(all_results, key=lambda x: x["fad"]):
        name = Path(r["generated_folder"]).name
        print(f"  {name:<28} {r['fad']:>10.4f}")
    print(sep)
    print("  ↓ FAD menor = mayor similitud con audio real\n")


if __name__ == "__main__":
    main()
