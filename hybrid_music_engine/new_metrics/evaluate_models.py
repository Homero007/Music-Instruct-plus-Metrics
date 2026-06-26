#!/usr/bin/env python3
"""
Evaluacion comparativa de modelos generativos de audio.

Calcula por modelo:
  - FAD: menor es mejor.
  - CLAP Score: mayor es mejor.
  - tempos_std: desviacion estandar de tempos estimados; menor indica tempo
    mas consistente dentro del bloque evaluado.

Tambien aplica prueba de Friedman por metrica cuando hay medidas repetidas
por bloques, por ejemplo:
  real/seed_0/classical, real/seed_1/classical, ...
  model_a/seed_0/classical, model_a/seed_1/classical, ...

Uso recomendado:
  python evaluate_models.py \
    --real data/real \
    --models musicgen=data/musicgen audioldm=data/audioldm magenta=data/magenta \
    --prompts prompts.csv \
    --output-dir metrics/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
METRIC_DIRECTIONS = {
    "fad": "min",
    "kad": "min",
    "clap_score": "max",
    "tempos_std": "min",
    "tempo_std_delta_vs_real": "min",
}

pd = None
friedmanchisquare = None
rankdata = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    folder: Path


def _require_import(package: str, install: str) -> None:
    try:
        __import__(package)
    except ImportError:
        print(f"[ERROR] Falta el paquete '{package}'. Instala con: {install}")
        sys.exit(1)


def load_analysis_dependencies() -> None:
    global pd, friedmanchisquare, rankdata

    _require_import("pandas", "pip install pandas")
    _require_import("scipy", "pip install scipy")
    _require_import("openpyxl", "pip install openpyxl")

    import pandas as pandas_module
    from scipy.stats import friedmanchisquare as friedmanchisquare_fn
    from scipy.stats import rankdata as rankdata_fn

    pd = pandas_module
    friedmanchisquare = friedmanchisquare_fn
    rankdata = rankdata_fn


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    seen: set[str] = set()
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
            name = name.strip()
            folder = Path(raw_path.strip())
        else:
            folder = Path(value.strip())
            name = folder.name

        if not name:
            raise ValueError(f"Nombre de modelo invalido en: {value}")
        if name in seen:
            raise ValueError(f"Modelo duplicado: {name}")
        if not folder.exists():
            raise FileNotFoundError(f"No existe la carpeta del modelo '{name}': {folder}")

        seen.add(name)
        models.append(ModelSpec(name=name, folder=folder.resolve()))
    return models


def list_audio_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def has_audio(folder: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        for p in folder.rglob("*")
    )


def discover_blocks(real_folder: Path, models: list[ModelSpec], block_mode: str) -> list[Path]:
    if block_mode == "none":
        return [Path(".")]

    real_dirs = {
        p.relative_to(real_folder)
        for p in real_folder.rglob("*")
        if p.is_dir() and has_audio(p)
    }

    common = []
    for rel in sorted(real_dirs):
        if all((model.folder / rel).exists() and has_audio(model.folder / rel) for model in models):
            common.append(rel)

    return common or [Path(".")]


def safe_name(value: str | Path) -> str:
    text = str(value).replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "root"


def load_prompts(prompts_path: Path | None) -> list[str] | dict[str, str] | None:
    if prompts_path is None:
        return None
    if not prompts_path.exists():
        raise FileNotFoundError(f"No existe el archivo de prompts: {prompts_path}")

    suffix = prompts_path.suffix.lower()
    if suffix == ".txt":
        return [
            line.strip()
            for line in prompts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if suffix == ".json":
        with prompts_path.open(encoding="utf-8") as f:
            data = json.load(f)
        prompt_map: dict[str, str] = {}
        for item in data:
            audio = item.get("audio") or item.get("audio_path") or item.get("file")
            text = item.get("text") or item.get("prompt") or item.get("caption")
            if audio and text:
                add_prompt_keys(prompt_map, Path(audio), str(text))
        return prompt_map

    prompt_map = {}
    with prompts_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            audio_col, text_col = row[0].strip(), row[1].strip()
            if i == 0 and audio_col.lower() in {"audio", "audio_path", "file", "path"}:
                continue
            add_prompt_keys(prompt_map, Path(audio_col), text_col)
    return prompt_map


def add_prompt_keys(prompt_map: dict[str, str], audio_path: Path, prompt: str) -> None:
    prompt_map[str(audio_path)] = prompt
    prompt_map[audio_path.as_posix()] = prompt
    prompt_map[audio_path.name] = prompt
    prompt_map[audio_path.stem] = prompt


def build_pairs_for_block(
    model_root: Path,
    block_folder: Path,
    prompts: list[str] | dict[str, str] | None,
) -> list[tuple[Path, str]]:
    if prompts is None:
        return []

    files = list_audio_files(block_folder)
    if isinstance(prompts, list):
        all_files = list_audio_files(model_root)
        if len(all_files) != len(prompts):
            raise ValueError(
                "Con prompts .txt, la cantidad de lineas debe coincidir con "
                f"los audios del modelo. {model_root.name}: {len(all_files)} audios, "
                f"{len(prompts)} prompts."
            )
        by_file = {file.resolve(): prompts[i] for i, file in enumerate(all_files)}
        return [(file, by_file[file.resolve()]) for file in files]

    pairs: list[tuple[Path, str]] = []
    for file in files:
        candidates = [
            str(file.resolve()),
            str(file),
            file.as_posix(),
            str(file.relative_to(model_root)) if file.is_relative_to(model_root) else "",
            file.relative_to(model_root).as_posix() if file.is_relative_to(model_root) else "",
            file.name,
            file.stem,
        ]
        prompt = next((prompts[key] for key in candidates if key in prompts), None)
        if prompt is not None:
            pairs.append((file, prompt))
    return pairs


def estimate_tempos(audio_folder: Path) -> tuple[float, float, int]:
    _require_import("librosa", "pip install librosa")
    import librosa

    tempos: list[float] = []
    for audio_path in list_audio_files(audio_folder):
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_float = float(np.asarray(tempo).reshape(-1)[0])
            if math.isfinite(tempo_float) and tempo_float > 0:
                tempos.append(tempo_float)
        except Exception as exc:
            print(f"[WARN] No se pudo estimar tempo en {audio_path.name}: {exc}")

    if not tempos:
        return float("nan"), float("nan"), 0
    return float(np.mean(tempos)), float(np.std(tempos, ddof=1)) if len(tempos) > 1 else 0.0, len(tempos)


def run_evaluation(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_folder = args.real.resolve()
    if not real_folder.exists():
        raise FileNotFoundError(f"No existe carpeta real: {real_folder}")

    models = parse_model_specs(args.models)
    prompts = load_prompts(args.prompts)
    blocks = discover_blocks(real_folder, models, args.block_mode)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Configuracion ===")
    print(f"Real:       {real_folder}")
    print(f"Modelos:    {', '.join(m.name for m in models)}")
    print(f"Bloques:    {len(blocks)} ({', '.join(str(b) for b in blocks[:5])}{'...' if len(blocks) > 5 else ''})")
    print(f"Metricas:   {', '.join(args.metrics)}")
    print(f"Salida:     {output_dir}\n")

    rows: list[dict] = []
    real_tempo_cache: dict[Path, tuple[float, float, int]] = {}

    for block in blocks:
        real_block = real_folder / block
        real_tempo_cache[block] = estimate_tempos(real_block) if "tempos_std" in args.metrics else (np.nan, np.nan, 0)

        for model in models:
            gen_block = model.folder / block
            row = {
                "block": str(block),
                "model": model.name,
                "real_folder": str(real_block),
                "generated_folder": str(gen_block),
            }

            print(f"--- {model.name} | bloque={block} ---")

            if "fad" in args.metrics:
                from fad_score import compute_fad

                fad_out = output_dir / "cache" / "fad" / model.name / safe_name(block)
                result = compute_fad(
                    real_folder=real_block,
                    gen_folder=gen_block,
                    extractor_name=args.fad_extractor,
                    extractor_kwargs={"device": args.device} if args.fad_extractor in {"vggish", "clap", "pann"} else {},
                    output_dir=fad_out,
                    use_cache=not args.no_cache,
                )
                row["fad"] = result["fad"]

            if "kad" in args.metrics:
                from kad_score import compute_kad

                kad_out = output_dir / "cache" / "kad" / model.name / safe_name(block)
                result = compute_kad(
                    real_folder=real_block,
                    gen_folder=gen_block,
                    extractor_name=args.fad_extractor,
                    extractor_kwargs={"device": args.device} if args.fad_extractor in {"vggish", "clap", "pann"} else {},
                    output_dir=kad_out,
                )
                row["kad"] = result["kad"]

            if "clap" in args.metrics:
                pairs = build_pairs_for_block(model.folder, gen_block, prompts)
                if pairs:
                    from clap_score import compute_clap_score

                    clap_out = output_dir / "cache" / "clap" / model.name / safe_name(block)
                    result = compute_clap_score(
                        pairs=pairs,
                        model_variant=args.clap_model,
                        device=args.device,
                        output_dir=clap_out,
                        use_cache=not args.no_cache,
                        run_label=f"{model.name}_{safe_name(block)}",
                    )
                    row["clap_score"] = result["clap_score"]
                    row["clap_std"] = result["clap_std"]
                    row["clap_n_pairs"] = result["n_pairs"]
                else:
                    row["clap_score"] = np.nan
                    row["clap_std"] = np.nan
                    row["clap_n_pairs"] = 0
                    print("[WARN] CLAP omitido: no hay prompts emparejables para este bloque.")

            if "tempos_std" in args.metrics:
                real_tempo_mean, real_tempo_std, real_tempo_n = real_tempo_cache[block]
                tempo_mean, tempo_std, tempo_n = estimate_tempos(gen_block)
                row["real_tempo_mean"] = real_tempo_mean
                row["real_tempo_std"] = real_tempo_std
                row["real_tempo_n"] = real_tempo_n
                row["tempo_mean"] = tempo_mean
                row["tempos_std"] = tempo_std
                row["tempo_n"] = tempo_n
                row["tempo_std_delta_vs_real"] = abs(tempo_std - real_tempo_std)

            rows.append(row)

    scores = pd.DataFrame(rows)
    ranking = build_ranking(scores)
    friedman = run_friedman_tests(scores, [m.name for m in models])
    return scores, ranking, friedman


def build_ranking(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in scores.groupby("model", sort=False):
        row = {"model": model, "n_blocks": group["block"].nunique()}
        for metric in METRIC_DIRECTIONS:
            if metric in group.columns:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)

    ranking = pd.DataFrame(rows)
    rank_cols = []
    for metric, direction in METRIC_DIRECTIONS.items():
        col = f"{metric}_mean"
        if col not in ranking.columns:
            continue
        rank_col = f"rank_{metric}"
        ranking[rank_col] = ranking[col].rank(ascending=(direction == "min"), method="average")
        rank_cols.append(rank_col)

    if rank_cols:
        ranking["overall_rank"] = ranking[rank_cols].mean(axis=1)
        ranking = ranking.sort_values(["overall_rank", "model"])
    return ranking.reset_index(drop=True)


def run_friedman_tests(scores: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    results = []
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in scores.columns:
            continue

        pivot = scores.pivot_table(index="block", columns="model", values=metric, aggfunc="mean")
        pivot = pivot.reindex(columns=model_names).dropna()
        n_blocks, n_models = pivot.shape

        if n_models < 3 or n_blocks < 2:
            results.append({
                "metric": metric,
                "n_blocks": n_blocks,
                "n_models": n_models,
                "chi_square": np.nan,
                "p_value": np.nan,
                "kendall_w": np.nan,
                "significant_alpha_0_05": False,
                "note": "Friedman requiere al menos 3 modelos y 2 bloques completos.",
            })
            continue

        arrays = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
        stat, p_value = friedmanchisquare(*arrays)
        kendall_w = float(stat / (n_blocks * (n_models - 1)))

        values = pivot.to_numpy(dtype=float)
        rank_input = values if direction == "min" else -values
        ranks = np.vstack([rankdata(row, method="average") for row in rank_input])
        mean_ranks = dict(zip(pivot.columns, ranks.mean(axis=0)))

        results.append({
            "metric": metric,
            "n_blocks": n_blocks,
            "n_models": n_models,
            "chi_square": float(stat),
            "p_value": float(p_value),
            "kendall_w": kendall_w,
            "significant_alpha_0_05": bool(p_value < 0.05),
            "best_mean_rank_model": min(mean_ranks, key=mean_ranks.get),
            "mean_ranks_json": json.dumps(mean_ranks, ensure_ascii=False),
            "note": "Diferencia significativa" if p_value < 0.05 else "Sin evidencia suficiente de diferencia significativa",
        })

    return pd.DataFrame(results)


def _import_companions():
    """Importa stats_tests y visualizations como hermanos o como paquete."""
    try:
        from . import stats_tests, visualizations  # type: ignore
    except ImportError:
        import stats_tests  # type: ignore
        import visualizations  # type: ignore
    return stats_tests, visualizations


def run_kruskal_tests(scores: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """
    Kruskal-Wallis entre modelos por métrica (muestras = valores por bloque),
    con post-hoc de Dunn (Bonferroni) cuando es significativo. Complementa a
    Friedman: Friedman asume medidas repetidas por bloque; Kruskal-Wallis trata
    los grupos como independientes y no asume normalidad (Fase 5 del protocolo).
    """
    stats_tests, _ = _import_companions()
    results = []
    for metric in METRIC_DIRECTIONS:
        if metric not in scores.columns:
            continue
        groups = {
            model: scores.loc[scores["model"] == model, metric].dropna().tolist()
            for model in model_names
        }
        groups = {name: vals for name, vals in groups.items() if vals}
        if len(groups) < 2:
            continue
        analysis = stats_tests.analyze_metric(groups, metric_name=metric)
        kw = analysis["kruskal_wallis"]
        dunn = analysis.get("posthoc_dunn")
        significant_pairs = (
            [
                f"{c['model_a']} vs {c['model_b']} (p={c['p_adjusted']:.4g})"
                for c in dunn["comparisons"] if c["significant"]
            ]
            if dunn else []
        )
        results.append({
            "metric": metric,
            "n_groups": kw["n_groups"],
            "h_statistic": kw["h_statistic"],
            "p_value": kw["p_value"],
            "dof": kw["dof"],
            "significant_alpha_0_05": kw["significant"],
            "dunn_significant_pairs": "; ".join(significant_pairs) if significant_pairs else "",
            "dunn_json": json.dumps(dunn, ensure_ascii=False) if dunn else "",
            "note": kw["note"],
        })
    return pd.DataFrame(results)


def save_outputs(
    scores: pd.DataFrame,
    ranking: pd.DataFrame,
    friedman: pd.DataFrame,
    output_dir: Path,
    kruskal: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if kruskal is None:
        model_names = list(dict.fromkeys(scores["model"].tolist())) if "model" in scores.columns else []
        kruskal = run_kruskal_tests(scores, model_names)

    plot_paths = create_plots(scores, ranking, friedman, plots_dir)
    workbook_path = output_dir / "evaluacion_modelos.xlsx"

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="ranking", index=False)
        scores.to_excel(writer, sheet_name="scores_por_bloque", index=False)
        friedman.to_excel(writer, sheet_name="friedman", index=False)
        if kruskal is not None and not kruskal.empty:
            kruskal.to_excel(writer, sheet_name="kruskal_wallis", index=False)

        summary = pd.DataFrame({
            "campo": [
                "interpretacion_fad",
                "interpretacion_clap",
                "interpretacion_tempos_std",
                "friedman",
                "kruskal_wallis",
                "dunn_posthoc",
            ],
            "valor": [
                "Menor FAD = distribucion de audio generado mas cercana a real.",
                "Mayor CLAP Score = mayor alineacion texto-audio.",
                "Menor tempos_std = tempos mas consistentes dentro del bloque.",
                "p < 0.05 sugiere diferencias significativas (medidas repetidas por bloque).",
                "p < 0.05 sugiere diferencias entre modelos sin asumir normalidad.",
                "Comparaciones por pares con correccion de Bonferroni cuando Kruskal-Wallis es significativo.",
            ],
        })
        summary.to_excel(writer, sheet_name="resumen", index=False)

        workbook = writer.book
        for sheet in workbook.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(length + 2, 12), 55)

        if plot_paths:
            from openpyxl.drawing.image import Image
            ws = workbook.create_sheet("graficas")
            row = 1
            for title, path in plot_paths.items():
                ws.cell(row=row, column=1, value=title)
                img = Image(str(path))
                img.width = 850
                img.height = 420
                ws.add_image(img, f"A{row + 1}")
                row += 24

    return {"xlsx": workbook_path, **plot_paths}


def create_plots(
    scores: pd.DataFrame,
    ranking: pd.DataFrame,
    friedman: pd.DataFrame,
    plots_dir: Path,
) -> dict[str, Path]:
    _require_import("matplotlib", "pip install matplotlib")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}

    if "overall_rank" in ranking.columns and not ranking.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordered = ranking.sort_values("overall_rank", ascending=True)
        ax.bar(ordered["model"], ordered["overall_rank"], color="#2563EB")
        ax.set_title("Ranking global por promedio de rangos")
        ax.set_ylabel("Rango promedio (menor = mejor)")
        ax.set_xlabel("Modelo")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plots_dir / "ranking_global.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["ranking_global"] = path

    available_metrics = [m for m in METRIC_DIRECTIONS if m in scores.columns]
    if available_metrics:
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, available_metrics):
            data = [
                group[metric].dropna().to_numpy(dtype=float)
                for _, group in scores.groupby("model", sort=False)
            ]
            labels = list(scores.groupby("model", sort=False).groups.keys())
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_title(f"Distribucion por bloques: {metric}")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plots_dir / "metricas_por_bloque.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["metricas_por_bloque"] = path

    if not friedman.empty and "p_value" in friedman.columns and friedman["p_value"].notna().any():
        data = friedman.dropna(subset=["p_value"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(data["metric"], data["p_value"], color="#0F766E")
        ax.axhline(0.05, color="#DC2626", linestyle="--", label="alpha = 0.05")
        ax.set_yscale("log")
        ax.set_title("Prueba de Friedman por metrica")
        ax.set_ylabel("p-value (escala log)")
        ax.set_xlabel("Metrica")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plots_dir / "friedman_pvalues.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["friedman_pvalues"] = path

    paths.update(_comparison_plots(scores, ranking, plots_dir))
    return paths


def _comparison_plots(scores: pd.DataFrame, ranking: pd.DataFrame, plots_dir: Path) -> dict[str, Path]:
    """
    Gráficas comparativas de la Fase 5 (radar, heatmap, dispersión FAD vs CLAP)
    delegando en el módulo visualizations. Cada una es independiente y se omite
    con aviso si faltan datos; nunca interrumpe el resto.
    """
    _, visualizations = _import_companions()
    paths: dict[str, Path] = {}

    # 1) Radar de métricas normalizadas (1 = mejor) a partir de las medias.
    metric_means = [m for m in METRIC_DIRECTIONS if f"{m}_mean" in ranking.columns]
    if not ranking.empty and len(metric_means) >= 2 and ranking["model"].nunique() >= 2:
        try:
            metrics_by_model = {
                str(row["model"]): {
                    m: float(row[f"{m}_mean"])
                    for m in metric_means
                    if pd.notna(row[f"{m}_mean"])
                }
                for _, row in ranking.iterrows()
            }
            directions = {m: (METRIC_DIRECTIONS[m] == "max") for m in metric_means}
            paths["radar"] = visualizations.radar_chart(
                metrics_by_model, directions, plots_dir / "radar_modelos.png"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Radar omitido: {exc}")

    # 2) Heatmap CLAP por modelo × bloque (= género cuando los bloques son géneros).
    if "clap_score" in scores.columns and scores["clap_score"].notna().any():
        try:
            pivot = scores.pivot_table(
                index="model", columns="block", values="clap_score", aggfunc="mean"
            )
            matrix = {
                str(model): {
                    str(block): float(val)
                    for block, val in row.items() if pd.notna(val)
                }
                for model, row in pivot.iterrows()
            }
            if matrix:
                paths["heatmap_clap"] = visualizations.heatmap_by_genre(
                    matrix, plots_dir / "heatmap_clap_modelo_genero.png"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Heatmap omitido: {exc}")

    # 3) Dispersión FAD vs CLAP (un punto por modelo).
    if not ranking.empty and "fad_mean" in ranking.columns and "clap_score_mean" in ranking.columns:
        try:
            points = {
                str(row["model"]): (float(row["fad_mean"]), float(row["clap_score_mean"]))
                for _, row in ranking.iterrows()
                if pd.notna(row["fad_mean"]) and pd.notna(row["clap_score_mean"])
            }
            if len(points) >= 1:
                paths["scatter_fad_clap"] = visualizations.scatter_fad_vs_clap(
                    points, plots_dir / "dispersion_fad_vs_clap.png"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Dispersión omitida: {exc}")

    return paths


def print_math_results(ranking: pd.DataFrame, friedman: pd.DataFrame) -> None:
    print("\n=== Ranking ordenado ===")
    if ranking.empty:
        print("No hay ranking disponible.")
    else:
        cols = [c for c in ["model", "overall_rank", "fad_mean", "kad_mean", "clap_score_mean", "tempos_std_mean", "tempo_std_delta_vs_real_mean"] if c in ranking.columns]
        print(ranking[cols].to_string(index=False))

    print("\n=== Prueba de Friedman ===")
    if friedman.empty:
        print("No se evaluaron metricas para Friedman.")
        return

    for _, row in friedman.iterrows():
        metric = row["metric"]
        if pd.isna(row["chi_square"]):
            print(f"{metric}: {row['note']}")
            continue
        print(
            f"{metric}: chi2_F({int(row['n_models']) - 1}) = {row['chi_square']:.6f}, "
            f"p = {row['p_value']:.6g}, W = {row['kendall_w']:.4f}. {row['note']}."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalua modelos con FAD, CLAP, tempos_std y Friedman.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real", type=Path, required=True, help="Carpeta con audio real/de referencia.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Modelos como nombre=ruta o solo ruta. Requiere 3+ modelos para Friedman.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["fad", "kad", "clap", "tempos_std"],
        default=["fad", "kad", "clap", "tempos_std"],
        help="Metricas a calcular.",
    )
    parser.add_argument("--prompts", type=Path, default=None, help="Prompts .txt, .csv o .json para CLAP.")
    parser.add_argument("--output-dir", type=Path, default=Path("metrics/evaluation"), help="Carpeta de salida.")
    parser.add_argument("--block-mode", choices=["auto", "none"], default="auto", help="Como crear medidas repetidas.")
    parser.add_argument("--fad-extractor", choices=["vggish", "pann", "clap", "mel"], default="vggish", help="Extractor para FAD.")
    parser.add_argument("--clap-model", choices=["clap", "clap-music"], default="clap", help="Variante CLAP.")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps.")
    parser.add_argument("--no-cache", action="store_true", help="Recalcular embeddings aunque exista cache.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if "clap" in args.metrics and args.prompts is None:
        print("[WARN] Pediste CLAP pero no pasaste --prompts. CLAP quedara omitido.")

    load_analysis_dependencies()
    scores, ranking, friedman = run_evaluation(args)
    output_paths = save_outputs(scores, ranking, friedman, args.output_dir.resolve())
    print_math_results(ranking, friedman)

    print("\n=== Archivos generados ===")
    print(f"Excel: {output_paths['xlsx']}")
    for key, path in output_paths.items():
        if key != "xlsx":
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
