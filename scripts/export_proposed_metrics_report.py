#!/usr/bin/env python3
"""Exporta el reporte CSV con las métricas propuestas del benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_text(value: str, *, scale: float = 1.0) -> str:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return ""
    return f"{float(value) * scale:.6f}"


def _training_epochs(training_csv: Path | None) -> int | str:
    if not training_csv or not training_csv.exists():
        return ""
    rows = _read_csv(training_csv)
    if not rows:
        return ""
    return max(int(row["epoch"]) for row in rows if row.get("epoch"))


def _load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _test_count(testset_csv: Path | None) -> int | str:
    if not testset_csv or not testset_csv.exists():
        return ""
    return len(_read_csv(testset_csv))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_csv", type=Path, default=Path("results/set_level_metrics.csv"))
    parser.add_argument("--clip", dest="clip_csv", type=Path, default=Path("results/clip_level_metrics.csv"))
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("results/transformer_200ep/training_metrics.csv"),
        help="CSV de entrenamiento usado para documentar épocas registradas.",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=None,
        help="summary.json del entrenamiento. Por defecto se busca junto al training CSV.",
    )
    parser.add_argument("--testset-csv", type=Path, default=Path("testset_metadata.csv"))
    parser.add_argument("--train-samples", default="", help="Número de muestras usadas en entrenamiento si no está en summary.json.")
    parser.add_argument("--run-name", default="", help="Nombre de corrida para agrupar train/test.")
    parser.add_argument("--train-command", default="", help="Comando usado para ejecutar entrenamiento.")
    parser.add_argument("--test-command", default="", help="Comando usado para generar este reporte/test.")
    parser.add_argument("--out", type=Path, default=Path("results/proposed_metrics_report.csv"))
    args = parser.parse_args()

    set_rows = _read_csv(args.set_csv)
    clip_rows = _read_csv(args.clip_csv)
    ternas_by_model = Counter(row["model"] for row in clip_rows if row.get("model"))
    epochs_registered = _training_epochs(args.training_csv)
    training_summary = args.training_summary or args.training_csv.parent / "summary.json"
    summary = _load_json(training_summary)
    run_name = args.run_name or args.training_csv.parent.name
    train_samples = summary.get("num_samples", args.train_samples)
    train_command = args.train_command or (
        f"python train_colab.py --epochs {epochs_registered} --batch-size 8 "
        f"--save-every 10 --checkpoint-dir {args.training_csv.parent / 'checkpoints'} "
        f"--results-dir {args.training_csv.parent}"
    )
    test_command = args.test_command or (
        f"python scripts/export_proposed_metrics_report.py --run-name {run_name} "
        f"--training-csv {args.training_csv} --out {args.out}"
    )

    rows: list[dict[str, str | int]] = [
        {
            "run_name": run_name,
            "split": "train",
            "record_name": f"{run_name}__train",
            "model": "SimpleTransformer",
            "epochs_registered": epochs_registered,
            "training_metrics_csv": str(args.training_csv.resolve()) if args.training_csv.exists() else "",
            "training_summary_json": str(training_summary.resolve()) if training_summary.exists() else "",
            "train_samples": train_samples,
            "test_samples": _test_count(args.testset_csv),
            "final_train_loss": _float_text(str(summary.get("final_loss", ""))) if summary.get("final_loss") is not None else "",
            "best_train_loss": _float_text(str(summary.get("best_loss", ""))) if summary.get("best_loss") is not None else "",
            "training_time_minutes": _float_text(str(summary.get("training_time_minutes", ""))) if summary.get("training_time_minutes") is not None else "",
            "ternas_evaluated": "",
            "meets_min_30_ternas": "",
            "fad_vggish": "",
            "fad_pann": "",
            "clap_score_mean": "",
            "clap_score_std": "",
            "compliance_rate_clap_gt_025": "",
            "compliance_percent_clap_gt_025": "",
            "kld": "",
            "kad": "",
            "source_set_csv": "",
            "source_clip_csv": "",
            "command": train_command,
        }
    ]
    for row in set_rows:
        model = row["model"]
        ternas = ternas_by_model.get(model, 0)
        compliance_rate = _float_text(row.get("pct_clap_above_025", ""), scale=0.01)
        rows.append(
            {
                "run_name": run_name,
                "split": "test",
                "record_name": f"{run_name}__test__{model}",
                "model": model,
                "epochs_registered": epochs_registered,
                "training_metrics_csv": str(args.training_csv.resolve()) if args.training_csv.exists() else "",
                "training_summary_json": str(training_summary.resolve()) if training_summary.exists() else "",
                "train_samples": train_samples,
                "test_samples": _test_count(args.testset_csv),
                "final_train_loss": _float_text(str(summary.get("final_loss", ""))) if summary.get("final_loss") is not None else "",
                "best_train_loss": _float_text(str(summary.get("best_loss", ""))) if summary.get("best_loss") is not None else "",
                "training_time_minutes": _float_text(str(summary.get("training_time_minutes", ""))) if summary.get("training_time_minutes") is not None else "",
                "ternas_evaluated": ternas,
                "meets_min_30_ternas": "yes" if ternas >= 30 else "no",
                "fad_vggish": _float_text(row.get("fad_vggish", "")),
                "fad_pann": _float_text(row.get("fad_pann", "")),
                "clap_score_mean": _float_text(row.get("mean_clap", "")),
                "clap_score_std": _float_text(row.get("std_clap", "")),
                "compliance_rate_clap_gt_025": compliance_rate,
                "compliance_percent_clap_gt_025": _float_text(row.get("pct_clap_above_025", "")),
                "kld": _float_text(row.get("kld", "")),
                "kad": _float_text(row.get("kad", "")),
                "source_set_csv": str(args.set_csv.resolve()),
                "source_clip_csv": str(args.clip_csv.resolve()),
                "command": test_command,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "source_set_csv": str(args.set_csv.resolve()),
                "source_clip_csv": str(args.clip_csv.resolve()),
                "run_name": run_name,
                "training_csv": str(args.training_csv.resolve()) if args.training_csv.exists() else "",
                "training_summary": str(training_summary.resolve()) if training_summary.exists() else "",
                "testset_csv": str(args.testset_csv.resolve()) if args.testset_csv.exists() else "",
                "commands": {
                    "train": train_command,
                    "test_report": test_command,
                },
                "metric_definitions": {
                    "fad_vggish": "Frechet Audio Distance con extractor VGGish; menor es mejor.",
                    "fad_pann": "Frechet Audio Distance con extractor PANN; menor es mejor.",
                    "clap_score_mean": "Media de similitud texto-audio CLAP por modelo; mayor es mejor.",
                    "compliance_rate_clap_gt_025": "Tasa de clips con CLAP-score > 0.25.",
                    "ternas_evaluated": "Número de ternas/filas clip-level evaluadas por modelo.",
                    "split": "train registra la corrida de entrenamiento; test registra métricas propuestas por modelo.",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"CSV escrito: {args.out.resolve()}")
    for row in rows:
        if row["split"] == "train":
            print(f"- train {row['run_name']}: epochs={row['epochs_registered']}, loss={row['final_train_loss']}")
        else:
            print(
                f"- test {row['model']}: ternas={row['ternas_evaluated']}, "
                f"FAD={row['fad_vggish']}, CLAP={row['clap_score_mean']}, "
                f"cumplimiento={row['compliance_rate_clap_gt_025']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
