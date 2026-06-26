"""
Tests del reward model con torch.

Estrategia: generamos pares sintéticos donde "preferred" tiene tempo en [110,130]
y density alta, y "rejected" tiene tempo extremo y density baja. Si el modelo
puede separarlos con accuracy alta, el pipeline está vivo de extremo a extremo.

Se saltan si torch no está disponible.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch    # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _skip_if_no_torch():
    if not HAS_TORCH:
        print("torch no instalado — pruebas omitidas")
        sys.exit(0)


def _synthetic_pairs(n: int = 200, seed: int = 42):
    from hybrid_music_engine.reward_model.dataset import PreferencePair
    import random
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        # Preferido: tempo cómodo + density media-alta + más notas
        pref = {
            "tempo_bpm": rng.uniform(100, 130),
            "note_density": rng.uniform(2.5, 5.0),
            "mean_velocity": rng.uniform(60, 90),
            "syncopation": rng.uniform(0.05, 0.3),
            "duration_seconds": 30.0,
            "num_notes": rng.randint(80, 200),
        }
        # Rechazado: tempo extremo + density muy baja
        rej = {
            "tempo_bpm": rng.choice([rng.uniform(40, 60), rng.uniform(180, 220)]),
            "note_density": rng.uniform(0.0, 1.0),
            "mean_velocity": rng.uniform(20, 50),
            "syncopation": rng.uniform(0.5, 1.0),
            "duration_seconds": 30.0,
            "num_notes": rng.randint(5, 30),
        }
        pairs.append(PreferencePair(pref, rej, weight=1.0, source="synthetic"))
    return pairs


def test_features_can_distinguish_synthetic():
    """Sanity: el FeatureSchema separa preferidos de rechazados en una columna."""
    from hybrid_music_engine.reward_model.features import FeatureSchema
    pairs = _synthetic_pairs(50)
    schema = FeatureSchema.fit([p.preferred for p in pairs] + [p.rejected for p in pairs])
    pref_X = schema.vectorize_batch(p.preferred for p in pairs)
    rej_X = schema.vectorize_batch(p.rejected for p in pairs)
    names = schema.feature_names
    # La densidad de notas debe ser estrictamente mayor en preferidos por construcción.
    idx = names.index("note_density")
    assert pref_X[:, idx].mean() > rej_X[:, idx].mean()


def test_train_reaches_high_accuracy():
    """Entrenamiento end-to-end: con datos sintéticos limpios debería alcanzar >85% accuracy."""
    from hybrid_music_engine.reward_model.train import TrainConfig, train_reward_model

    pairs = _synthetic_pairs(400, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        report = train_reward_model(
            pairs, out_dir=Path(tmp),
            cfg=TrainConfig(epochs=30, batch_size=64, lr=2e-3, patience=8, seed=1, val_fraction=0.2),
        )
        assert report["best_val_acc"] > 0.85, f"acc bajo: {report['best_val_acc']}"
        assert Path(report["model_path"]).exists()
        assert Path(report["schema_path"]).exists()


def test_score_consistent_with_training_signal():
    """Después de entrenar, scorer(preferred) > scorer(rejected) en promedio."""
    from hybrid_music_engine.reward_model.train import TrainConfig, train_reward_model
    from hybrid_music_engine.reward_model.score import RewardScorer

    pairs = _synthetic_pairs(400, seed=2)
    with tempfile.TemporaryDirectory() as tmp:
        report = train_reward_model(
            pairs, out_dir=Path(tmp),
            cfg=TrainConfig(epochs=25, lr=2e-3, patience=6, seed=2, val_fraction=0.2),
        )
        scorer = RewardScorer(report["model_path"], report["schema_path"])
        test_pairs = _synthetic_pairs(60, seed=99)
        pref_scores = [scorer.score(p.preferred) for p in test_pairs]
        rej_scores = [scorer.score(p.rejected) for p in test_pairs]
        wins = sum(1 for a, b in zip(pref_scores, rej_scores) if a > b)
        assert wins / len(test_pairs) > 0.85, f"win rate bajo: {wins/len(test_pairs)}"


def test_rerank_reorders_synthetic_ranking():
    """rerank_ranking_file levanta candidatas 'mejores' al top en un ranking sintético."""
    import json
    from hybrid_music_engine.reward_model.train import TrainConfig, train_reward_model
    from hybrid_music_engine.reward_model.score import RewardScorer
    from hybrid_music_engine.reward_model.rerank import rerank_ranking_file

    pairs = _synthetic_pairs(400, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = train_reward_model(
            pairs, out_dir=tmp_path / "model",
            cfg=TrainConfig(epochs=25, lr=2e-3, patience=6, seed=3, val_fraction=0.2),
        )
        scorer = RewardScorer(report["model_path"], report["schema_path"])

        # Ranking sintético: 4 candidatas, dos buenas (índice 1, 3), dos malas.
        test_pairs = _synthetic_pairs(2, seed=10)
        ranking = {
            "candidates": [
                {"name": "cand-A-mala", "metrics": test_pairs[0].rejected, "score": 0.8},
                {"name": "cand-B-buena", "metrics": test_pairs[0].preferred, "score": 0.6},
                {"name": "cand-C-mala", "metrics": test_pairs[1].rejected, "score": 0.7},
                {"name": "cand-D-buena", "metrics": test_pairs[1].preferred, "score": 0.5},
            ],
        }
        ranking_path = tmp_path / "ranking.json"
        ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
        rep = rerank_ranking_file(ranking_path, scorer, alpha=1.0)
        out = json.loads(rep.output_path.read_text(encoding="utf-8"))
        top_names = [c["name"] for c in out["candidates"][:2]]
        # Las dos buenas deben quedar en el top-2.
        assert set(top_names) == {"cand-B-buena", "cand-D-buena"}, f"top-2 = {top_names}"
        # Sanity: el reporte debe haber computado Spearman entre reward y heurístico original.
        assert "rerank" in out and "spearman_vs_original" in out["rerank"]


if __name__ == "__main__":
    _skip_if_no_torch()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("TODAS LAS PRUEBAS DEL REWARD MODEL PASARON")
