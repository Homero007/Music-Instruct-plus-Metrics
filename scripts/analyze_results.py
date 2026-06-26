#!/usr/bin/env python3
"""analyze_results.py — Genera los 3 productos del benchmark a partir de los CSV.

Uso:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --results results/
  python scripts/analyze_results.py --set results/set_level_metrics.csv \
                                     --clip results/clip_level_metrics.csv \
                                     --out results/

Produce en out_dir:
  comparison_table.csv / .md  — tabla FAD/CLAP/KLD/KAD con rangos
  clap_kruskal_dunn.json      — Kruskal-Wallis + Dunn por género
  radar_models.png            — radar normalizado 0-1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Carpeta con los CSV (default: results/)")
    parser.add_argument("--set", dest="set_csv", default=None, help="Ruta explícita a set_level_metrics.csv")
    parser.add_argument("--clip", dest="clip_csv", default=None, help="Ruta explícita a clip_level_metrics.csv")
    parser.add_argument("--out", default=None, help="Carpeta de salida (default: misma que los CSV)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    results_dir = root / args.results if not Path(args.results).is_absolute() else Path(args.results)

    set_csv = Path(args.set_csv) if args.set_csv else results_dir / "set_level_metrics.csv"
    clip_csv = Path(args.clip_csv) if args.clip_csv else results_dir / "clip_level_metrics.csv"
    out_dir = Path(args.out) if args.out else results_dir

    if not set_csv.exists():
        print(f"Error: no existe {set_csv}")
        print("Genera los datos con:  python scripts/make_synthetic_results.py")
        sys.exit(1)
    if not clip_csv.exists():
        print(f"Error: no existe {clip_csv}")
        sys.exit(1)

    sys.path.insert(0, str(root))
    from hybrid_music_engine.new_metrics.benchmark_analysis import run_analysis

    print(f"Leyendo {set_csv}")
    print(f"Leyendo {clip_csv}")
    result = run_analysis(set_csv, clip_csv, out_dir)

    table = result["table"]
    print(f"\n✓ Tabla comparativa  →  {table['md']}")
    print(f"                          {table['csv']}")
    print(f"\n✓ Kruskal-Wallis+Dunn →  {result['stats']['path']}")
    print(f"\n✓ Radar normalizado   →  {result['radar']['png']}")

    print("\nResumen de la tabla:")
    print(f"  {'Modelo':<22} {'FAD':>7} {'CLAP':>12} {'KLD':>7} {'KAD':>7} {'Rango':>6}")
    for row in table["rows"]:
        print(
            f"  {row['model']:<22} {row['fad_vggish']:>7.3f} "
            f"{row['clap_mean']:>6.3f}±{row['clap_std']:.3f} "
            f"{row['kld']:>7.3f} {row['kad']:>7.3f} {row['overall_rank']:>6.2f}"
        )


if __name__ == "__main__":
    main()
