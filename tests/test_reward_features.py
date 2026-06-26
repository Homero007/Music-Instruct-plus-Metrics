"""Tests de features (sin torch)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_music_engine.reward_model.features import FeatureSchema, flatten_metrics  # noqa: E402


def test_flatten_handles_metrics_wrapper():
    out = flatten_metrics({"metrics": {"tempo": 120, "density": 0.5}})
    assert out == {"tempo": 120, "density": 0.5}


def test_flatten_handles_nested_one_level():
    out = flatten_metrics({"global": {"tempo": 120}, "extra": {"density": 0.5}})
    assert out["tempo"] == 120 and out["density"] == 0.5


def test_alias_resolution_picks_first_match():
    schema = FeatureSchema()
    # Mismo valor accesible por dos aliases distintos
    v1 = schema.vectorize({"tempo": 120, "density": 0.5})
    v2 = schema.vectorize({"tempo_bpm": 120, "note_density": 0.5})
    assert v1.shape == v2.shape == (schema.dim,)
    # Las columnas que coinciden por alias deben tener el mismo valor.
    names = schema.feature_names
    idx_tempo = names.index("tempo_bpm")
    idx_density = names.index("note_density")
    assert v1[idx_tempo] == v2[idx_tempo] == 120
    assert v1[idx_density] == v2[idx_density] == 0.5


def test_missing_fields_are_flagged():
    schema = FeatureSchema()
    v = schema.vectorize({"tempo": 120})    # falta casi todo
    names = schema.feature_names
    # bandera missing del tempo = 0 (presente)
    assert v[names.index("tempo_bpm__missing")] == 0.0
    # bandera missing de syncopation = 1 (ausente)
    assert v[names.index("syncopation__missing")] == 1.0


def test_pitch_classes_vector_to_stats():
    schema = FeatureSchema()
    v = schema.vectorize({"pitch_classes": [3, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0]})
    names = schema.feature_names
    # mean, std, max, entropy_norm — todos deben ser finitos y > 0 para max/mean
    assert v[names.index("pitch_classes__mean")] > 0
    assert v[names.index("pitch_classes__max")] == 3.0
    assert 0.0 < v[names.index("pitch_classes__entropy_norm")] < 1.0
    assert v[names.index("pitch_classes__missing")] == 0.0


def test_pitch_classes_as_dict():
    schema = FeatureSchema()
    v_list = schema.vectorize({"pitch_classes": [1, 2, 3]})
    v_dict = schema.vectorize({"pitch_classes": {"C": 1, "C#": 2, "D": 3}})
    names = schema.feature_names
    for stat in ("mean", "std", "max", "entropy_norm"):
        assert v_list[names.index(f"pitch_classes__{stat}")] == v_dict[names.index(f"pitch_classes__{stat}")]


def test_fit_computes_medians_and_zscore():
    schema = FeatureSchema.fit([
        {"tempo": 100, "density": 0.4, "syncopation": 0.1},
        {"tempo": 120, "density": 0.5, "syncopation": 0.2},
        {"tempo": 140, "density": 0.6, "syncopation": 0.3},
    ])
    assert schema.medians["tempo_bpm"] == 120
    # z-score: con mean ajustado, la media de la columna debe ser ~0
    X = schema.vectorize_batch([{"tempo": 100}, {"tempo": 120}, {"tempo": 140}])
    Xs = schema.standardize(X)
    names = schema.feature_names
    idx = names.index("tempo_bpm")
    assert abs(float(Xs[:, idx].mean())) < 1e-4


def test_roundtrip_json(tmp_path: Path | None = None):
    import tempfile
    schema = FeatureSchema.fit([{"tempo": 100, "density": 0.5}])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = Path(f.name)
    schema.to_json(path)
    loaded = FeatureSchema.from_json(path)
    assert loaded.dim == schema.dim
    assert loaded.feature_names == schema.feature_names
    v_a = schema.vectorize({"tempo": 95}); v_b = loaded.vectorize({"tempo": 95})
    assert (v_a == v_b).all()
    path.unlink()


def test_non_finite_values_treated_as_missing():
    schema = FeatureSchema()
    v = schema.vectorize({"tempo": float("inf"), "density": float("nan")})
    names = schema.feature_names
    assert v[names.index("tempo_bpm__missing")] == 1.0
    assert v[names.index("note_density__missing")] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("TODAS LAS PRUEBAS DE FEATURES PASARON")
