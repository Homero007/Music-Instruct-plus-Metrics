"""Pruebas de las métricas del benchmark (benchmark_analysis.py).

Cubre: tabla comparativa + ranking, radar normalizado [0,1], Kruskal-Wallis +
Dunn (global y por género) y el payload completo desde CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("scipy")
from hybrid_music_engine.new_metrics import benchmark_analysis as ba

# name, fad, clap, kld, kad  (MusicGen-medium es el mejor en las 4 métricas)
MODELS = [
    ("MusicGen-medium", 1.42, 0.318, 0.18, 0.021),
    ("MusicGen-small", 1.95, 0.273, 0.27, 0.034),
    ("AudioLDM2", 2.71, 0.221, 0.41, 0.052),
    ("Stable-Audio-Open", 3.15, 0.210, 0.52, 0.061),
]
GENRES = ["jazz", "rock", "classical", "pop"]


def _set_rows():
    return [
        {
            "model": n, "fad_vggish": f"{fad}", "fad_pann": f"{fad * 0.9}",
            "mean_clap": f"{clap}", "std_clap": "0.05", "pct_clap_above_025": "50",
            "kld": f"{kld}", "kad": f"{kad}",
        }
        for (n, fad, clap, kld, kad) in MODELS
    ]


def _clip_rows():
    import random
    rng = random.Random(0)
    rows = []
    for (n, _fad, clap, kld, _kad) in MODELS:
        for g in GENRES:
            for i in range(8):
                rows.append({
                    "model": n, "clip_id": f"{n}_{g}_{i}", "genre": g, "tempo_bpm": "120",
                    "clap_score": f"{max(0.0, rng.gauss(clap, 0.02)):.5f}",
                    "passt_kld": f"{kld:.4f}",
                })
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_compute_table_ranks_best_first():
    table = ba.compute_table(_set_rows())
    assert {"rows", "models"} <= set(table)  # puede incluir methodological_notes, etc.
    assert len(table["rows"]) == 4
    first = table["rows"][0]
    assert first["model"] == "MusicGen-medium"
    assert first["overall_rank"] == 1.0
    for row in table["rows"]:
        for key in ("model", "fad_vggish", "clap_mean", "kld", "kad", "overall_rank"):
            assert key in row


def test_compute_radar_normalized_in_unit_range():
    radar = ba.compute_radar(_set_rows())
    assert radar["models"]
    assert set(radar["metrics"]) == {"fad", "clap", "kld", "kad"}
    for per_model in radar["normalized"].values():
        for value in per_model.values():
            assert 0.0 <= value <= 1.0
        assert max(per_model.values()) == pytest.approx(1.0)
        assert min(per_model.values()) == pytest.approx(0.0)
    # MusicGen-medium (mejor) normaliza a 1.0 en FAD (que se minimiza).
    assert radar["normalized"]["fad"]["MusicGen-medium"] == pytest.approx(1.0)


def test_compute_clap_stats_kruskal_dunn():
    stats = ba.compute_clap_stats(_clip_rows())
    assert stats["metric"] == "clap_score"
    overall = stats["overall"]
    assert overall["h_statistic"] is not None
    assert isinstance(overall["p_value"], float)
    assert isinstance(overall["significant"], bool)
    assert isinstance(overall["dunn"], list)
    assert set(stats["by_genre"]) == set(GENRES)


def test_benchmark_payload_from_csv(tmp_path):
    set_csv = tmp_path / "set_level_metrics.csv"
    clip_csv = tmp_path / "clip_level_metrics.csv"
    _write_csv(set_csv, _set_rows())
    _write_csv(clip_csv, _clip_rows())

    payload = ba.benchmark_payload(set_csv, clip_csv)
    assert {"table", "stats", "radar", "sources"} <= set(payload)
    assert payload["table"]["rows"][0]["model"] == "MusicGen-medium"
    assert payload["radar"]["models"]
    assert payload["stats"]["overall"]["h_statistic"] is not None


def test_benchmark_payload_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ba.benchmark_payload(tmp_path / "nope_set.csv", tmp_path / "nope_clip.csv")


# ── Casos límite y escritura de productos ───────────────────────────────────

def test_table_handles_missing_metric():
    rows = _set_rows()
    rows[0] = {k: v for k, v in rows[0].items() if k != "kad"}  # falta una métrica
    table = ba.compute_table(rows)
    assert len(table["rows"]) == 4
    assert all("overall_rank" in r for r in table["rows"])


def test_radar_all_equal_normalizes_to_neutral():
    # Empate total en una métrica (span == 0) → valor neutro 0.5 para todos.
    rows = [
        {"model": f"M{i}", "fad_vggish": "2.0", "mean_clap": "0.3", "std_clap": "0.05",
         "kld": "0.3", "kad": "0.04"}
        for i in range(3)
    ]
    radar = ba.compute_radar(rows)
    for per_model in radar["normalized"].values():
        assert all(v == 0.5 for v in per_model.values())


def test_clap_stats_few_models_no_kruskal():
    rows = [{"model": n, "genre": "jazz", "clap_score": "0.3"} for n in ("A", "B") for _ in range(6)]
    stats = ba.compute_clap_stats(rows)
    assert stats["overall"]["h_statistic"] is None
    assert stats["overall"]["significant"] is False


def test_clap_stats_detects_significant_difference():
    rows = []
    for n, base in (("A", 0.1), ("B", 0.5), ("C", 0.9)):
        for i in range(15):
            rows.append({"model": n, "genre": "jazz", "clap_score": f"{base + 0.001 * i}"})
    stats = ba.compute_clap_stats(rows)
    assert stats["overall"]["h_statistic"] is not None
    assert stats["overall"]["significant"] is True


def test_run_analysis_writes_all_products(tmp_path):
    pytest.importorskip("matplotlib")
    set_csv = tmp_path / "set_level_metrics.csv"
    clip_csv = tmp_path / "clip_level_metrics.csv"
    _write_csv(set_csv, _set_rows())
    _write_csv(clip_csv, _clip_rows())
    out = tmp_path / "out"
    result = ba.run_analysis(set_csv, clip_csv, out)
    assert Path(result["table"]["csv"]).exists()
    assert Path(result["table"]["md"]).exists()
    assert Path(result["stats"]["path"]).exists()
    assert Path(result["radar"]["png"]).exists()
