#!/usr/bin/env python3
"""
pipeline.py — Orquestador de la etapa de MÉTRICAS NUEVAS (post-render).

Toma la salida de audio de una generación (WAV/MP3, idealmente con subcarpetas
por género) y produce un PAQUETE DE GRÁFICOS + un report.json, en su propia
subcarpeta data/new_metrics/<run_id>/. No toca el reward model ni los rankings.

Métricas que orquesta (cada una opcional según insumos disponibles):
  • FAD + tempo  -> vía evaluate_models (gráficos por bloque/género + xlsx).
  • CLAP         -> vía evaluate_models si hay --prompts (necesita checkpoint CLAP).
  • t-SNE palabras -> vía words_tsne si hay --t5-dir (gráfico 3 géneros + por género).
  • KLD          -> si se pasan probabilidades real/generadas (.npy/.csv/.json).

Diseño:
  • Reutiliza los scripts existentes SIN modificarlos (se importan como hermanos).
  • Degrada con elegancia: si falta un insumo o una dependencia, esa métrica se
    omite con aviso y el resto continúa.
  • La prioridad es GRÁFICOS: todo .png se consolida en plots/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("new_metrics.pipeline")

AUDIO_METRICS_DEFAULT = ["fad", "tempos_std"]


def _new_run_id(name: str) -> str:
    return f"newmetrics_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _collect_pngs(src_dir: Path, plots_dir: Path, prefix: str) -> list[Path]:
    """Copia todos los .png de src_dir (recursivo) a plots_dir con prefijo."""
    out: list[Path] = []
    for png in sorted(src_dir.rglob("*.png")):
        dst = plots_dir / f"{prefix}{png.name}"
        shutil.copy2(png, dst)
        out.append(dst)
    return out


# ── Sub-etapas ────────────────────────────────────────────────────────────────

def _run_audio_metrics(
    generated_root: Path,
    real_root: Path | None,
    out_dir: Path,
    metrics: Sequence[str],
    fad_extractor: str,
    clap_model: str,
    device: str,
    prompts: Path | None,
    model_label: str,
) -> dict:
    """FAD/CLAP/tempo vía evaluate_models. Devuelve {ranking, plots, xlsx}."""
    import evaluate_models as em

    if real_root is None:
        # Sin referencia real no se puede FAD. Conservamos solo lo reference-free.
        metrics = [m for m in metrics if m not in {"fad"}]
        if not metrics:
            log.warning("Sin --real y sin métricas reference-free: se omite audio.")
            return {}

    audio_out = out_dir / "audio"
    ns = SimpleNamespace(
        real=real_root if real_root is not None else generated_root,
        models=[f"{model_label}={generated_root}"],
        metrics=list(metrics),
        prompts=prompts,
        output_dir=audio_out,
        block_mode="auto",
        fad_extractor=fad_extractor,
        clap_model=clap_model,
        device=device,
        no_cache=False,
    )

    em.load_analysis_dependencies()
    scores, ranking, friedman = em.run_evaluation(ns)
    paths = em.save_outputs(scores, ranking, friedman, audio_out)

    return {
        "ranking": ranking.to_dict(orient="records") if hasattr(ranking, "to_dict") else [],
        "scores": scores.to_dict(orient="records") if hasattr(scores, "to_dict") else [],
        "xlsx": str(paths.get("xlsx")) if paths.get("xlsx") else None,
        "plot_src": audio_out / "plots",
    }


def _run_words_tsne(t5_dir: Path, out_dir: Path, knn: int, seed: int) -> dict:
    """t-SNE SOLO de palabras vía words_tsne. Devuelve {plot_src, summary}."""
    import words_tsne as wt

    items = wt.load_t5_dir(t5_dir)
    if not items:
        log.warning("t5-dir sin .npz: se omite t-SNE.")
        return {}
    tsne_out = out_dir / "tsne"
    produced = wt.build_word_maps(items, tsne_out, seed=seed, knn_k=knn)
    return {"plot_src": tsne_out, "summary": str(produced.get("summary"))}


def _run_kld(
    real_probs: Path,
    gen_probs: Path,
    out_dir: Path,
    plots_dir: Path,
) -> dict:
    """KLD(real||gen) desde probabilidades + un gráfico de las distribuciones."""
    import kld_metric as kld
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr = kld.cargar_predicciones(real_probs)
    pg = kld.cargar_predicciones(gen_probs)
    p = kld.promediar_predicciones(pr)
    q = kld.promediar_predicciones(pg)
    value = kld.calcular_kld_desde_distribuciones(p, q)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(p))
    w = 0.4
    ax.bar(x - w / 2, p, width=w, label="real (P)", color="#2563EB")
    ax.bar(x + w / 2, q, width=w, label="generado (Q)", color="#DC2626")
    ax.set_title(f"KLD(P‖Q) = {value:.4f} — distribuciones por clase")
    ax.set_xlabel("clase")
    ax.set_ylabel("probabilidad media")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = plots_dir / "kld_distribuciones.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"kld": float(value), "plot": str(path)}


# ── Orquestador ───────────────────────────────────────────────────────────────

def run_new_metrics(
    generated_root: str | Path,
    out_dir: str | Path | None = None,
    *,
    real_root: str | Path | None = None,
    t5_dir: str | Path | None = None,
    prompts: str | Path | None = None,
    metrics: Sequence[str] | None = None,
    fad_extractor: str = "mel",
    clap_model: str = "clap",
    device: str = "cpu",
    model_label: str = "candidatas",
    kld_real_probs: str | Path | None = None,
    kld_gen_probs: str | Path | None = None,
    knn: int = 6,
    seed: int = 42,
    run_name: str = "run",
) -> dict:
    """
    Ejecuta la etapa de métricas nuevas y deja un paquete de gráficos + report.json.

    Devuelve el report (dict) con rutas de gráficos y resumen numérico.
    """
    generated_root = Path(generated_root)
    if not generated_root.exists():
        raise FileNotFoundError(f"No existe generated_root: {generated_root}")

    out_dir = Path(out_dir) if out_dir else Path("data/new_metrics") / _new_run_id(run_name)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "stage": "new_metrics",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generated_root": str(generated_root.resolve()),
        "real_root": str(Path(real_root).resolve()) if real_root else None,
        "out_dir": str(out_dir.resolve()),
        "plots": [],
        "audio": {},
        "tsne": {},
        "kld": {},
        "note": "Etapa independiente del reward model; salida orientada a gráficos.",
    }

    # 1) Métricas de audio (FAD/CLAP/tempo) -> gráficos por bloque/género.
    audio_metrics = list(metrics) if metrics is not None else list(AUDIO_METRICS_DEFAULT)
    try:
        audio = _run_audio_metrics(
            generated_root, Path(real_root) if real_root else None, out_dir,
            audio_metrics, fad_extractor, clap_model, device, prompts, model_label,
        )
        if audio:
            report["audio"] = {k: v for k, v in audio.items() if k != "plot_src"}
            if audio.get("plot_src") and Path(audio["plot_src"]).exists():
                report["plots"] += [str(p) for p in _collect_pngs(Path(audio["plot_src"]), plots_dir, "audio_")]
    except Exception as exc:  # noqa: BLE001
        log.warning("Audio metrics fallaron: %s", exc)
        report["audio"] = {"error": str(exc)}

    # 2) t-SNE SOLO de palabras.
    if t5_dir:
        try:
            tsne = _run_words_tsne(Path(t5_dir), out_dir, knn=knn, seed=seed)
            report["tsne"] = {k: v for k, v in tsne.items() if k != "plot_src"}
            if tsne.get("plot_src") and Path(tsne["plot_src"]).exists():
                report["plots"] += [str(p) for p in _collect_pngs(Path(tsne["plot_src"]), plots_dir, "tsne_")]
        except Exception as exc:  # noqa: BLE001
            log.warning("t-SNE falló: %s", exc)
            report["tsne"] = {"error": str(exc)}

    # 3) KLD (si hay probabilidades).
    if kld_real_probs and kld_gen_probs:
        try:
            report["kld"] = _run_kld(Path(kld_real_probs), Path(kld_gen_probs), out_dir, plots_dir)
            if report["kld"].get("plot"):
                report["plots"].append(report["kld"]["plot"])
        except Exception as exc:  # noqa: BLE001
            log.warning("KLD falló: %s", exc)
            report["kld"] = {"error": str(exc)}

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    log.info("Report -> %s", report_path)
    log.info("Gráficos: %d en %s", len(report["plots"]), plots_dir)
    return report
