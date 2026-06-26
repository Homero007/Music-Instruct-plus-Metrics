#!/usr/bin/env python3
"""
cli.py — Interfaz de línea de comandos de la etapa de métricas nuevas.

Ejemplos:
    # FAD (mel) + tempo + t-SNE de palabras
    python -m new_metrics.cli \
        --generated data/renders/RENDER_ID \
        --real data/datasets/jamendo/delivery_jamendo_150 \
        --t5-dir data/encodings_v2/t5_seq \
        --fad-extractor mel \
        --out data/new_metrics/demo

    # Añadiendo CLAP (requiere prompts y checkpoint CLAP) y KLD (probabilidades)
    python -m new_metrics.cli --generated ... --real ... \
        --metrics fad clap tempos_std --prompts prompts.csv \
        --kld-real real_probs.npy --kld-gen gen_probs.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_new_metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Etapa de métricas nuevas (post-render): gráficos de FAD/CLAP/KLD/t-SNE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--generated", type=Path, required=True,
                   help="Salida de audio a evaluar (idealmente subcarpetas por género).")
    p.add_argument("--real", type=Path, default=None, help="Audio real de referencia (FAD).")
    p.add_argument("--t5-dir", type=Path, default=None, help="encodings_v2/t5_seq (t-SNE palabras).")
    p.add_argument("--prompts", type=Path, default=None, help="Prompts .txt/.csv/.json (CLAP).")
    p.add_argument("--metrics", nargs="+", default=["fad", "tempos_std"],
                   choices=["fad", "clap", "tempos_std"])
    p.add_argument("--fad-extractor", choices=["vggish", "clap", "mel"], default="mel")
    p.add_argument("--clap-model", choices=["clap", "clap-music"], default="clap")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-label", default="candidatas", help="Etiqueta del set evaluado.")
    p.add_argument("--kld-real", type=Path, default=None, help="Probabilidades reales (.npy/.csv/.json).")
    p.add_argument("--kld-gen", type=Path, default=None, help="Probabilidades generadas.")
    p.add_argument("--knn", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None, help="Carpeta de salida (data/new_metrics/<id>).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    report = run_new_metrics(
        generated_root=args.generated,
        out_dir=args.out,
        real_root=args.real,
        t5_dir=args.t5_dir,
        prompts=args.prompts,
        metrics=args.metrics,
        fad_extractor=args.fad_extractor,
        clap_model=args.clap_model,
        device=args.device,
        model_label=args.model_label,
        kld_real_probs=args.kld_real,
        kld_gen_probs=args.kld_gen,
        knn=args.knn,
        seed=args.seed,
    )
    print("\n=== Métricas nuevas: gráficos generados ===")
    for path in report["plots"]:
        print(" ", path)
    print(f"\nReport: {report['report_path']}")


if __name__ == "__main__":
    main()
