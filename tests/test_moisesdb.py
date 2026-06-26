"""Pruebas de la integración con MoisesDB (stems reales).

Usa un fixture sintético con la estructura cruda de MoisesDB (subcarpetas por
categoría + data.json) para verificar el adaptador y la construcción de ternas
con stems reales, sin descargar el dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

sf = pytest.importorskip("soundfile")
import moisesdb_adapter as moises
import build_edit_triplets as bt


def _make_moisesdb(root: Path, rng) -> Path:
    track = root / "trackA"
    for cat, amp in [("drums", 0.8), ("bass", 0.2), ("vocals", 0.15), ("guitar", 0.1)]:
        d = track / cat
        d.mkdir(parents=True)
        sf.write(str(d / "a.wav"), rng.normal(0, amp, 32000).astype(np.float32), 32000)
    (track / "data.json").write_text(json.dumps({"genre": "rock", "song": "x"}), encoding="utf-8")
    return track


def test_list_tracks(tmp_path):
    _make_moisesdb(tmp_path, np.random.default_rng(0))
    tracks = moises.list_tracks(tmp_path)
    assert len(tracks) == 1 and tracks[0].name == "trackA"


def test_load_track_stems_taxonomy_and_canonical(tmp_path):
    _make_moisesdb(tmp_path, np.random.default_rng(0))
    rec = moises.load_track_stems(moises.list_tracks(tmp_path)[0])
    assert set(rec["stems"]) == {"drums", "bass", "vocals", "other"}
    assert rec["mixture"].shape[-1] == moises.TARGET_LEN
    for s in rec["stems"].values():
        assert s.shape[-1] == moises.TARGET_LEN
    # 'guitar' se mapeó a la capa 'other'.
    assert np.any(rec["stems"]["other"] != 0)


def test_track_genre(tmp_path):
    _make_moisesdb(tmp_path, np.random.default_rng(0))
    assert moises.track_genre(moises.list_tracks(tmp_path)[0]) == "rock"


def test_mixture_is_sum_of_stems(tmp_path):
    _make_moisesdb(tmp_path, np.random.default_rng(0))
    rec = moises.load_track_stems(moises.list_tracks(tmp_path)[0])
    assert np.allclose(rec["mixture"], sum(rec["stems"].values()), atol=1e-6)


def test_list_tracks_robust_to_provider_and_nesting(tmp_path):
    # Estructura real: <root>/<provider>/<track_id>/ con data.json y track-types anidados.
    rng = np.random.default_rng(0)
    track = tmp_path / "spotify" / "track-uuid-123"
    for cat, sub in [("drums", "kick"), ("vocals", "lead_vocal"), ("bass", "bass_guitar")]:
        d = track / cat / sub  # categoría -> track-type -> wav (anidado)
        d.mkdir(parents=True)
        sf.write(str(d / "x.wav"), rng.normal(0, 0.3, 16000).astype(np.float32), 16000)
    (track / "data.json").write_text(json.dumps({"genre": "pop"}), encoding="utf-8")

    tracks = moises.list_tracks(tmp_path)
    assert tracks == [track]  # encontrada pese a la anidación por provider
    rec = moises.load_track_stems(track)
    # los wavs anidados (track-types) se sumaron y resamplearon a 32 kHz/10 s
    assert rec["mixture"].shape[-1] == moises.TARGET_LEN
    assert np.any(rec["stems"]["drums"] != 0)


def test_best_offset_segment_selection():
    sr = 1000
    win_s = 2
    y = np.full(10 * sr, 0.01)
    y[5 * sr:7 * sr] = 1.0  # burst de energía en el centro
    # energy: la ventana cae sobre el burst (no en el inicio silencioso).
    off = moises._best_offset(y, sr, win_s, "energy")
    assert 4 * sr <= off <= 6 * sr
    # start / middle deterministas.
    assert moises._best_offset(y, sr, win_s, "start") == 0
    assert moises._best_offset(y, sr, win_s, "middle") == (len(y) - win_s * sr) // 2


def test_build_triplets_moisesdb_real_stems(tmp_path):
    src_root = tmp_path / "moises"
    _make_moisesdb(src_root, np.random.default_rng(0))
    tracks = moises.list_tracks(src_root)
    out = tmp_path / "out"

    rows = bt.build_triplets_moisesdb(tracks, ["remove_drums", "quieter"], out, no_bpm=True)
    by_op = {r["operation"]: r for r in rows}

    # remove_drums usa stems REALES (no Demucs).
    assert by_op["remove_drums"]["stems_origin"] == "real"
    assert "moisesdb_real_stems" in by_op["remove_drums"]["target_method"]
    # quitar la batería (capa fuerte) reduce el RMS total.
    assert float(by_op["remove_drums"]["target_attr"]) < float(by_op["remove_drums"]["source_attr"])
    assert by_op["remove_drums"]["genre"] == "rock"
    # DSP no usa stems.
    assert by_op["quieter"]["stems_origin"] == "n/a"
    # fuente (mezcla) y objetivos escritos a disco.
    for r in rows:
        assert Path(r["source_path"]).exists()
        assert Path(r["target_path"]).exists()
