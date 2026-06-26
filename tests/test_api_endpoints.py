"""Pruebas de los endpoints del backend con el TestClient de FastAPI.

Cubre: /api/health, /api/jobs/generate-pretrained (validación + creación de job
mockeando el dispatcher para no lanzar el subprocess real) y
/api/metrics/benchmark (con CSVs hermético y caso de archivos faltantes).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("scipy")

from fastapi.testclient import TestClient

from hybrid_music_engine.api.main import app

client = TestClient(app)

MODELS = [
    ("MusicGen-medium", 1.42, 0.318, 0.18, 0.021),
    ("MusicGen-small", 1.95, 0.273, 0.27, 0.034),
    ("AudioLDM2", 2.71, 0.221, 0.41, 0.052),
    ("Stable-Audio-Open", 3.15, 0.210, 0.52, 0.061),
]
GENRES = ["jazz", "rock", "classical"]


def _w(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_set(path):
    _w(path, [
        {"model": n, "fad_vggish": f"{fad}", "fad_pann": f"{fad}", "mean_clap": f"{clap}",
         "std_clap": "0.05", "pct_clap_above_025": "50", "kld": f"{kld}", "kad": f"{kad}"}
        for (n, fad, clap, kld, kad) in MODELS
    ])


def _write_clip(path):
    import random
    rng = random.Random(0)
    rows = []
    for (n, _f, clap, kld, _k) in MODELS:
        for g in GENRES:
            for i in range(6):
                rows.append({
                    "model": n, "clip_id": f"{n}_{g}_{i}", "genre": g, "tempo_bpm": "120",
                    "clap_score": f"{max(0.0, rng.gauss(clap, 0.02)):.5f}", "passt_kld": f"{kld}",
                })
    _w(path, rows)


# ── /api/health ──────────────────────────────────────────────────────────────

def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "project_root" in body


# ── /api/jobs/generate-pretrained ────────────────────────────────────────────

def test_generate_pretrained_rejects_invalid_model():
    r = client.post("/api/jobs/generate-pretrained", json={"model_name": "no-existe"})
    assert r.status_code == 422


def test_generate_pretrained_requires_model_name():
    r = client.post("/api/jobs/generate-pretrained", json={})
    assert r.status_code == 422  # pydantic: falta model_name


def test_generate_pretrained_valid_creates_job(monkeypatch):
    # Mockea el dispatcher para NO lanzar el subprocess real de generación.
    import hybrid_music_engine.jobs.dispatcher as disp
    monkeypatch.setattr(disp, "dispatch_generate_pretrained", lambda job_id, model: "stub-mode")
    r = client.post("/api/jobs/generate-pretrained", json={"model_name": "musicgen-small"})
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body and body["mode"] == "stub-mode"


# ── /api/metrics/benchmark ───────────────────────────────────────────────────

def test_benchmark_with_csvs(tmp_path):
    set_csv = tmp_path / "set.csv"
    clip_csv = tmp_path / "clip.csv"
    _write_set(set_csv)
    _write_clip(clip_csv)
    r = client.get("/api/metrics/benchmark",
                   params={"set_level": str(set_csv), "clip_level": str(clip_csv)})
    assert r.status_code == 200
    body = r.json()
    assert {"table", "stats", "radar"} <= set(body)
    assert body["table"]["rows"][0]["model"] == "MusicGen-medium"


def test_benchmark_missing_files_returns_400(tmp_path):
    r = client.get("/api/metrics/benchmark",
                   params={"set_level": str(tmp_path / "nope.csv"),
                           "clip_level": str(tmp_path / "nope2.csv")})
    assert r.status_code == 400
