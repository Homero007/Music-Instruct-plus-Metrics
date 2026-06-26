"""Pruebas del CLI end-to-end del benchmark de edición (run_edit_benchmark.py).

Construye un banco de ternas DSP real y verifica las anclas: la reconstrucción
preserva pero no opera; un modelo "perfecto" (salida=objetivo) opera y queda
primero en el ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

import build_edit_triplets as bt
import run_edit_benchmark as reb
from hybrid_music_engine.new_metrics import edit_benchmark as eb


def _make_triplets(tmp_path):
    """Banco DSP pequeño desde varias fuentes sintéticas distintas."""
    out = tmp_path / "trip"
    sources = []
    rng = np.random.default_rng(0)
    for i in range(4):
        s = tmp_path / f"src{i}.wav"
        sf.write(str(s), rng.normal(0, 0.3, bt.TARGET_LEN).astype(np.float32), bt.TARGET_SR)
        sources.append(s)
    rows = bt.build_triplets(sources, ["quieter", "fade_out"], out, no_bpm=True)
    import csv
    with open(out / "edit_triplets.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    return manifest


def test_reconstruction_anchor(tmp_path):
    rows = _make_triplets(tmp_path)
    rec = reb.evaluate_model("reconstruction", rows, reb.reconstruction_output)
    # salida = fuente -> preserva perfecto y NO aplica la operación.
    # (tolerancia por el redondeo a 4 decimales de los atributos en el manifest)
    assert all(p == pytest.approx(1.0, abs=1e-6) for p in rec["content_preservation"])
    assert all(s < 0.05 for s in rec["operation_success"])


def test_perfect_model_operates(tmp_path):
    rows = _make_triplets(tmp_path)

    def perfect(row):
        return reb.load_audio(Path(row["target_path"]))

    perf = reb.evaluate_model("perfect", rows, perfect)
    rec = reb.evaluate_model("reconstruction", rows, reb.reconstruction_output)
    # salida = objetivo -> éxito de operación máximo.
    assert all(s > 0.95 for s in perf["operation_success"])
    # FAD a objetivo: el modelo perfecto (salida=objetivo) está más cerca que la
    # reconstrucción (salida=fuente != objetivo). (Magnitud absoluta no testeada:
    # el Fréchet necesita N >> dim para ser estable; aquí se testea la relación.)
    assert (eb.fad_to_target(perf["output_embs"], perf["target_embs"])
            < eb.fad_to_target(rec["output_embs"], rec["target_embs"]))


def test_oracle_anchor_brackets_reconstruction(tmp_path):
    rows = _make_triplets(tmp_path)
    oracle = reb.evaluate_model("oracle", rows, reb.oracle_output)
    recon = reb.evaluate_model("reconstruction", rows, reb.reconstruction_output)
    table = eb.compute_edit_table({"oracle": oracle, "reconstruction": recon})
    # El oráculo (salida=objetivo) es la cota superior -> primero.
    assert table["rows"][0]["model"] == "oracle"
    # Anclas: oráculo opera (~1), reconstrucción no (~0); reconstrucción preserva (1).
    by = {r["model"]: r for r in table["rows"]}
    assert by["oracle"]["operation_success"] > 0.95
    assert by["reconstruction"]["operation_success"] < 0.05
    assert by["reconstruction"]["content_preservation"] == pytest.approx(1.0, abs=1e-6)


def test_table_ranks_perfect_above_reconstruction(tmp_path):
    rows = _make_triplets(tmp_path)
    rec = reb.evaluate_model("reconstruction", rows, reb.reconstruction_output)
    perf = reb.evaluate_model("perfect", rows, lambda r: reb.load_audio(Path(r["target_path"])))
    table = eb.compute_edit_table({"reconstruction": rec, "perfect": perf})
    assert table["rows"][0]["model"] == "perfect"
    # CLAP no calculado (sin laion-clap) -> NaN, pero el ranking no se rompe.
    assert table["rows"][0]["clap_instruction"] != table["rows"][0]["clap_instruction"]  # NaN
