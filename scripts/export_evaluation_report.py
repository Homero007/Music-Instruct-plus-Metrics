#!/usr/bin/env python3
"""Exporta un CSV consolidado de evaluaciones formales.

El CSV resume FAD, CLAP-score, tasa de cumplimiento y conteo de ternas/pairs.
La tasa de cumplimiento se define como pares sin errores / pares totales.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _pair_rows(evaluation_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(evaluation_dir / "metrics" / "pair_metrics.json")
    return list(payload.get("pairs") or [])


def _summary(evaluation_dir: Path) -> dict[str, Any]:
    return _load_json(evaluation_dir / "metrics" / "summary.json")


def _row_for_evaluation(evaluation_dir: Path, *, min_pairs: int) -> dict[str, Any] | None:
    summary = _summary(evaluation_dir)
    pairs = _pair_rows(evaluation_dir)
    pair_count = len(pairs)
    if pair_count < min_pairs:
        return None

    valid_pairs = sum(1 for row in pairs if not row.get("errors"))
    compliance_rate = (valid_pairs / pair_count) if pair_count else None
    clap = summary.get("clap")
    clap_status = "ok" if clap is not None else "missing_dependency:laion-clap"

    return {
        "evaluation_id": summary.get("evaluation_id") or evaluation_dir.name,
        "evaluation_dir": str(evaluation_dir.resolve()),
        "pairs_total": pair_count,
        "pairs_valid": valid_pairs,
        "compliance_rate": _fmt(compliance_rate),
        "fad_mel": _fmt(summary.get("fad")),
        "clap_score": _fmt(clap),
        "clap_status": clap_status,
        "kld": _fmt(summary.get("kld")),
        "kad": _fmt(summary.get("kad")),
        "valid_midi_rate": _fmt(summary.get("valid_midi_rate")),
        "generated_audio": summary.get("generated_audio", ""),
        "generated_midis": summary.get("generated_midis", ""),
        "error_count": len(summary.get("errors") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluations-root",
        type=Path,
        default=Path("data/evaluations"),
        help="Carpeta con data/evaluations/<evaluation_id>.",
    )
    parser.add_argument(
        "--evaluation-id",
        action="append",
        default=[],
        help="ID específico a incluir. Se puede repetir. Si se omite, incluye todas.",
    )
    parser.add_argument("--min-pairs", type=int, default=30, help="Mínimo de ternas/pairs exigido.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/evaluation_system_metrics.csv"),
        help="CSV consolidado de salida.",
    )
    args = parser.parse_args()

    root = args.evaluations_root
    if args.evaluation_id:
        evaluation_dirs = [root / item for item in args.evaluation_id]
    else:
        evaluation_dirs = sorted(path for path in root.glob("*") if path.is_dir())

    rows = []
    for evaluation_dir in evaluation_dirs:
        row = _row_for_evaluation(evaluation_dir, min_pairs=args.min_pairs)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No hay evaluaciones con al menos {args.min_pairs} ternas en {root}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV escrito: {args.out.resolve()}")
    print(f"Evaluaciones incluidas: {len(rows)}")
    for row in rows:
        print(
            f"- {row['evaluation_id']}: pairs={row['pairs_total']}, "
            f"FAD={row['fad_mel']}, CLAP={row['clap_score'] or row['clap_status']}, "
            f"cumplimiento={row['compliance_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
