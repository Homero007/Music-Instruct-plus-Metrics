"""Pruebas del benchmark de EDICIÓN por instrucciones (edit_benchmark.py).

Valida las 5 métricas y sus casos ancla (reconstrucción, salida=objetivo) y que
la tabla rankee correctamente un editor real por encima de las líneas base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("scipy")
from hybrid_music_engine.new_metrics import edit_benchmark as eb


# ── operation_success: monotonía y anclas ───────────────────────────────────

def test_operation_success_reconstruction_is_zero():
    # salida = fuente -> no aplicó la operación.
    assert eb.operation_success(source_score=0.1, output_score=0.1, target_score=0.9) == 0.0


def test_operation_success_perfect_is_one():
    assert eb.operation_success(0.1, 0.9, 0.9) == pytest.approx(1.0)


def test_operation_success_partial_and_clipped():
    # mitad del camino
    assert eb.operation_success(0.0, 0.5, 1.0) == pytest.approx(0.5)
    # sobrepaso -> recortado a 1.0
    assert eb.operation_success(0.0, 1.5, 1.0) == 1.0
    # dirección equivocada -> recortado a 0.0
    assert eb.operation_success(0.0, -0.3, 1.0) == 0.0


def test_operation_success_remove_direction():
    # "quitar": objetivo < fuente; la salida que baja avanza hacia el objetivo.
    assert eb.operation_success(0.9, 0.5, 0.1) == pytest.approx(0.5)


def test_operation_success_no_change_required():
    # objetivo ~ fuente: éxito = qué tan cerca quedó de la fuente.
    assert eb.operation_success(0.5, 0.5, 0.5) == pytest.approx(1.0)
    assert eb.operation_success(0.5, 0.7, 0.5) == pytest.approx(0.8)


# ── content_preservation ────────────────────────────────────────────────────

def test_content_preservation_identical_is_one():
    v = np.array([0.2, -0.5, 0.7, 0.1])
    assert eb.content_preservation(v, v) == pytest.approx(1.0)


def test_content_preservation_more_similar_scores_higher():
    src = np.array([1.0, 0.0, 0.0])
    close = np.array([0.9, 0.1, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    assert eb.content_preservation(src, close) > eb.content_preservation(src, far)


def test_content_preservation_stems_only_unedited():
    rng = np.random.default_rng(0)
    drums = rng.normal(size=8)
    bass = rng.normal(size=8)
    voice = rng.normal(size=8)
    source = {"drums": drums, "bass": bass, "voice": voice}
    # La operación añade/cambia "drums"; bass y voice se preservan idénticos.
    output = {"drums": rng.normal(size=8), "bass": bass, "voice": voice}
    score = eb.content_preservation_stems(source, output, edited_keys={"drums"})
    assert score == pytest.approx(1.0)  # solo mide bass+voice, intactos


def test_content_preservation_stems_all_edited_is_nan():
    s = {"drums": np.ones(4)}
    o = {"drums": np.zeros(4)}
    assert np.isnan(eb.content_preservation_stems(s, o, edited_keys={"drums"}))


# ── fad_to_target y clap_instruction ────────────────────────────────────────

def test_fad_to_target_closer_set_scores_lower():
    rng = np.random.default_rng(0)
    target = rng.normal(0, 1, (50, 16))
    close = rng.normal(0.1, 1, (50, 16))
    far = rng.normal(3, 1.5, (50, 16))
    assert eb.fad_to_target(close, target) < eb.fad_to_target(far, target)


def test_clap_instruction_mean_cosine():
    a = np.array([1.0, 0.0])
    aligned = np.array([1.0, 0.0])
    orthogonal = np.array([0.0, 1.0])
    score = eb.clap_instruction_score([(a, aligned), (a, orthogonal)])
    assert score == pytest.approx(0.5)  # (1 + 0) / 2


# ── Tabla comparativa: un editor real supera a las líneas base ───────────────

def test_compute_edit_table_ranks_editor_above_baselines():
    rng = np.random.default_rng(0)
    target = rng.normal(0, 1, (40, 8))

    per_model = {
        # Editor bueno: salidas cercanas al objetivo, alto CLAP, preserva y opera.
        "Instruct-Editor": {
            "output_embs": target + rng.normal(0, 0.2, target.shape),
            "target_embs": target,
            "clap_scores": [0.45] * 40,
            "content_preservation": [0.9] * 40,
            "operation_success": [0.85] * 40,
        },
        # Reconstrucción: preserva perfecto pero NO aplica la operación.
        "Reconstruction": {
            "output_embs": rng.normal(0, 1, (40, 8)) + 2.0,
            "target_embs": target,
            "clap_scores": [0.15] * 40,
            "content_preservation": [1.0] * 40,
            "operation_success": [0.0] * 40,
        },
        # Solo texto: ignora la fuente, no preserva.
        "MusicGen-text-only": {
            "output_embs": rng.normal(0, 1, (40, 8)) + 1.0,
            "target_embs": target,
            "clap_scores": [0.30] * 40,
            "content_preservation": [0.2] * 40,
            "operation_success": [0.4] * 40,
        },
    }
    table = eb.compute_edit_table(per_model)
    assert set(table) == {"rows", "metrics", "labels"}
    assert table["rows"][0]["model"] == "Instruct-Editor"
    # Reconstrucción debe quedar mal en éxito de operación (su talón de Aquiles).
    recon = next(r for r in table["rows"] if r["model"] == "Reconstruction")
    assert recon["operation_success"] == 0.0
    assert recon["content_preservation"] == 1.0
