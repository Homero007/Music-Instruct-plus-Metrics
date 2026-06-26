"""
cli.py — Punto de entrada por línea de comandos.

Subcomandos:
  bootstrap   → construye pares automáticos real-vs-generado y los guarda
  train       → entrena un reward model (toma JSONL/JSON/CSV o un bootstrap.json)
  score       → puntúa un MIDI o un dict de métricas
  rerank      → re-rankea un ranking.json existente
  rerank-all  → re-rankea todos los ranking.json bajo data/ranked/
  evaluate    → reporta pairwise accuracy sobre un manifiesto

Ejecución:
  python -m instruct_music_engine.reward_model.cli <subcomando> --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("reward_model")


def _add_metrics_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metrics-mode", choices=["auto", "api", "local"], default="auto",
                   help="Cómo obtener métricas para los MIDI")
    p.add_argument("--api-url", default=None, help="URL de la API local (override de HYBRID_ENGINE_API_URL)")


def cmd_bootstrap(args) -> int:
    from .dataset import build_bootstrap_pairs
    from .metrics_provider import make_metrics_fn

    metrics_fn = make_metrics_fn(args.metrics_mode, api_url=args.api_url or "http://127.0.0.1:8100")
    pairs = build_bootstrap_pairs(
        real_dir=Path(args.real_dir),
        generated_dir=Path(args.generated_dir),
        metrics_fn=metrics_fn,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps({"preferred": p.preferred, "rejected": p.rejected,
                              "weight": p.weight, "source": p.source}) for p in pairs),
        encoding="utf-8",
    )
    log.info("Bootstrap escrito: %s (%d pares)", out_path, len(pairs))
    return 0


def cmd_train(args) -> int:
    from .dataset import read_preference_manifest, merge_pairs
    from .metrics_provider import make_metrics_fn
    from .features import FeatureSchema
    from .train import TrainConfig, train_reward_model

    metrics_fn = make_metrics_fn(args.metrics_mode, api_url=args.api_url or "http://127.0.0.1:8100")
    pairs = read_preference_manifest(Path(args.pairs), metrics_fn=metrics_fn)
    if args.extra_pairs:
        extra = read_preference_manifest(Path(args.extra_pairs), metrics_fn=metrics_fn)
        pairs = merge_pairs(pairs, extra)
    log.info("Total de pares: %d", len(pairs))

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_fraction=args.val_fraction, patience=args.patience,
        hidden_dims=tuple(args.hidden), dropout=args.dropout,
        seed=args.seed, device=args.device,
    )
    report = train_reward_model(pairs, out_dir=Path(args.output_dir), cfg=cfg)
    print(json.dumps(report, indent=2))
    return 0


def cmd_score(args) -> int:
    from .score import RewardScorer
    from .metrics_provider import make_metrics_fn

    scorer = RewardScorer(args.model, args.schema, device=args.device)
    if args.metrics_json:
        metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    elif args.midi:
        metrics_fn = make_metrics_fn(args.metrics_mode, api_url=args.api_url or "http://127.0.0.1:8100")
        metrics = metrics_fn(Path(args.midi))
    else:
        print("Pasa --midi o --metrics-json", file=sys.stderr); return 2
    print(json.dumps({"reward_score": scorer.score(metrics)}, indent=2))
    return 0


def cmd_rerank(args) -> int:
    from .score import RewardScorer
    from .rerank import rerank_ranking_file
    from .metrics_provider import make_metrics_fn

    scorer = RewardScorer(args.model, args.schema, device=args.device)
    metrics_fn = make_metrics_fn(args.metrics_mode, api_url=args.api_url or "http://127.0.0.1:8100")
    report = rerank_ranking_file(
        Path(args.ranking), scorer, alpha=args.alpha, metrics_fn=metrics_fn,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps({
        "ranking_path": str(report.ranking_path),
        "output_path": str(report.output_path),
        "n_candidates": report.n_candidates,
        "n_scored": report.n_scored,
        "n_missing_metrics": report.n_missing_metrics,
        "alpha": report.alpha,
        "spearman_vs_original": report.spearman_vs_original,
    }, indent=2))
    return 0


def cmd_rerank_all(args) -> int:
    from .score import RewardScorer
    from .rerank import rerank_directory
    from .metrics_provider import make_metrics_fn

    scorer = RewardScorer(args.model, args.schema, device=args.device)
    metrics_fn = make_metrics_fn(args.metrics_mode, api_url=args.api_url or "http://127.0.0.1:8100")
    reports = rerank_directory(Path(args.root), scorer, alpha=args.alpha, metrics_fn=metrics_fn)
    print(json.dumps([{
        "ranking_path": str(r.ranking_path), "output_path": str(r.output_path),
        "n_candidates": r.n_candidates, "n_scored": r.n_scored,
        "spearman_vs_original": r.spearman_vs_original,
    } for r in reports], indent=2))
    return 0


def cmd_evaluate(args) -> int:
    from .score import RewardScorer
    from .dataset import read_preference_manifest
    from .train import evaluate
    from .features import FeatureSchema
    from .metrics_provider import make_metrics_fn
    from .model import load_model

    metrics_fn = make_metrics_fn(args.metrics_mode, api_url=args.api_url or "http://127.0.0.1:8100")
    pairs = read_preference_manifest(Path(args.pairs), metrics_fn=metrics_fn)
    schema = FeatureSchema.from_json(args.schema)
    model = load_model(args.model, device=args.device)
    print(json.dumps(evaluate(model, pairs, schema, device=args.device), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reward_model")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("bootstrap", help="Construir pares automáticos real-vs-generado")
    pp.add_argument("--real-dir", required=True, help="Directorio con MIDIs reales (Jamendo procesado)")
    pp.add_argument("--generated-dir", required=True, help="Directorio con MIDIs generados (data/generated o data/ranked)")
    pp.add_argument("--output", required=True, help="Ruta de salida .jsonl")
    pp.add_argument("--max-pairs", type=int, default=2000)
    pp.add_argument("--seed", type=int, default=42)
    _add_metrics_args(pp)
    pp.set_defaults(func=cmd_bootstrap)

    pt = sub.add_parser("train", help="Entrenar reward model con pares de preferencia")
    pt.add_argument("--pairs", required=True, help=".jsonl/.json/.csv con preferred/rejected")
    pt.add_argument("--extra-pairs", default=None, help="Pares humanos extra (peso mayor)")
    pt.add_argument("--output-dir", required=True)
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--val-fraction", type=float, default=0.15)
    pt.add_argument("--patience", type=int, default=5)
    pt.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    pt.add_argument("--dropout", type=float, default=0.1)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--device", default="cpu")
    _add_metrics_args(pt)
    pt.set_defaults(func=cmd_train)

    ps = sub.add_parser("score", help="Puntúa un MIDI o un dict de métricas")
    ps.add_argument("--model", required=True)
    ps.add_argument("--schema", required=True)
    ps.add_argument("--midi", default=None)
    ps.add_argument("--metrics-json", default=None)
    ps.add_argument("--device", default="cpu")
    _add_metrics_args(ps)
    ps.set_defaults(func=cmd_score)

    pr = sub.add_parser("rerank", help="Re-rankear un ranking.json")
    pr.add_argument("--ranking", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--schema", required=True)
    pr.add_argument("--alpha", type=float, default=1.0, help="1=solo reward, 0=solo heurística")
    pr.add_argument("--output", default=None, help="ranking_reranked.json al lado por defecto")
    pr.add_argument("--device", default="cpu")
    _add_metrics_args(pr)
    pr.set_defaults(func=cmd_rerank)

    pa = sub.add_parser("rerank-all", help="Re-rankear todos los ranking.json bajo una raíz")
    pa.add_argument("--root", default="data/ranked")
    pa.add_argument("--model", required=True)
    pa.add_argument("--schema", required=True)
    pa.add_argument("--alpha", type=float, default=1.0)
    pa.add_argument("--device", default="cpu")
    _add_metrics_args(pa)
    pa.set_defaults(func=cmd_rerank_all)

    pe = sub.add_parser("evaluate", help="Pairwise accuracy sobre un manifiesto")
    pe.add_argument("--pairs", required=True)
    pe.add_argument("--model", required=True)
    pe.add_argument("--schema", required=True)
    pe.add_argument("--device", default="cpu")
    _add_metrics_args(pe)
    pe.set_defaults(func=cmd_evaluate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
