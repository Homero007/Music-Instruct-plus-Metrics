"""Pruebas del generador del banco de ternas de edición (build_edit_triplets.py).

Cubre el registro de operaciones, los transforms de DSP (deterministas), y un
end-to-end DSP que escribe objetivos + manifest con la metadolatía de
construcción transparente.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_edit_triplets as bt


# ── Registro de operaciones bien formado ─────────────────────────────────────

def test_operations_registry_wellformed():
    for name, op in bt.OPERATIONS.items():
        assert op["kind"] in {"dsp", "stem"}
        assert op["attr"] in bt.ATTR_FNS
        assert op["direction"] in {"increase", "decrease"}
        assert op["instructions"], f"{name} sin instrucciones"
        assert all(isinstance(s, str) and s for s in op["instructions"])
        if op["kind"] == "dsp":
            assert callable(op["build"])


def test_instruction_for_returns_template():
    rng = random.Random(0)
    instr = bt.instruction_for("quieter", rng)
    assert instr in bt.OPERATIONS["quieter"]["instructions"]


# ── Transforms de DSP ────────────────────────────────────────────────────────

def test_op_gain_scales_rms():
    y = np.random.default_rng(0).normal(size=1000)
    out = bt.op_gain(y, db=-6.0)
    factor = 10 ** (-6.0 / 20.0)
    assert bt.rms(out) == pytest.approx(bt.rms(y) * factor, rel=1e-6)


def test_op_fade_out_reduces_tail():
    y = np.ones(32000)
    out = bt.op_fade_out(y, sr=32000, seconds=1.0)
    assert bt.tail_energy(out, 32000, seconds=1.0) < bt.tail_energy(y, 32000, seconds=1.0)
    # el inicio se mantiene intacto
    assert out[0] == pytest.approx(1.0)


def test_fit_length_pads_and_trims():
    assert bt.fit_length(np.ones(10), n=20).shape[-1] == 20
    assert bt.fit_length(np.ones(30), n=20).shape[-1] == 20


def test_op_time_stretch_keeps_canonical_length():
    pytest.importorskip("librosa")
    y = np.random.default_rng(0).normal(size=bt.TARGET_LEN).astype(np.float32)
    out = bt.op_time_stretch(y, sr=bt.TARGET_SR, rate=0.82)
    assert out.shape[-1] == bt.TARGET_LEN  # recortado/rellenado a 10 s


# ── End-to-end (solo DSP, sin demucs ni librosa) ─────────────────────────────

def test_build_triplets_end_to_end(tmp_path):
    sf = pytest.importorskip("soundfile")
    # Fuente sintética de 10 s a 32 kHz.
    src = tmp_path / "src.wav"
    y = np.random.default_rng(1).normal(0, 0.3, bt.TARGET_LEN).astype(np.float32)
    sf.write(str(src), y, bt.TARGET_SR)

    out = tmp_path / "triplets"
    rows = bt.build_triplets([src], ["quieter", "fade_out"], out, no_bpm=True)

    assert len(rows) == 2
    manifest = out / "edit_triplets.csv"
    assert manifest.exists()

    with open(manifest) as f:
        data = list(csv.DictReader(f))
    cols = set(data[0].keys())
    assert {"id", "source_path", "target_path", "instruction", "operation",
            "kind", "edited_stems", "attr_name", "attr_direction",
            "source_attr", "target_attr", "target_method"} <= cols

    by_op = {r["operation"]: r for r in data}
    # quieter: el objetivo tiene menor RMS que la fuente (operación aplicada).
    assert float(by_op["quieter"]["target_attr"]) < float(by_op["quieter"]["source_attr"])
    # fade_out: menor energía de cola.
    assert float(by_op["fade_out"]["target_attr"]) < float(by_op["fade_out"]["source_attr"])
    # método de construcción del objetivo documentado.
    assert by_op["quieter"]["target_method"].startswith("dsp:")
    # los WAV objetivo existen y duran 10 s.
    for r in data:
        tgt = Path(r["target_path"])
        assert tgt.exists()
        info = sf.info(str(tgt))
        assert info.frames == bt.TARGET_LEN
