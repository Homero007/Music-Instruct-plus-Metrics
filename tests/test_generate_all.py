"""Pruebas de scripts/generate_all.py (generación con los 4 modelos).

No descarga pesos: cubre el post-proceso de audio (normalización, resample),
la detección de dispositivo y el RUTEO de main() hacia cada generador
(mockeando los generadores para no cargar transformers/diffusers).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
import generate_all as ga


# ── Post-proceso de audio ───────────────────────────────────────────────────

def test_peak_normalize_sets_target_peak():
    wav = torch.tensor([[0.1, -0.5, 0.25, 0.4]])
    out = ga.peak_normalize(wav, target_dbfs=-3.0)
    expected_peak = 10 ** (-3.0 / 20.0)
    assert out.abs().max().item() == pytest.approx(expected_peak, rel=1e-5)


def test_peak_normalize_silent_is_safe():
    wav = torch.zeros(1, 100)
    out = ga.peak_normalize(wav, target_dbfs=-3.0)
    assert torch.equal(out, wav)  # sin división por cero
    assert torch.isfinite(out).all()


def test_peak_normalize_preserves_shape():
    wav = torch.randn(2, 320)
    out = ga.peak_normalize(wav, target_dbfs=-1.0)
    assert out.shape == wav.shape


# ── Resample ────────────────────────────────────────────────────────────────

def test_resample_noop_same_sr():
    wav = torch.randn(1, 1000)
    out = ga.resample_if_needed(wav, 32000, 32000)
    assert torch.equal(out, wav)


def test_resample_changes_length():
    wav = torch.randn(1, 16000)  # 1 s @ 16 kHz
    out = ga.resample_if_needed(wav, 16000, 32000)
    assert out.shape[-1] == pytest.approx(32000, abs=64)  # ~2x al duplicar la tasa


# ── Device ──────────────────────────────────────────────────────────────────

def test_detect_device_valid():
    assert ga.detect_device() in {"mps", "cuda", "cpu"}


# ── main(): ruteo de modelos (sin cargar pesos) ─────────────────────────────

@pytest.fixture
def stub_generators(monkeypatch):
    """Reemplaza los 3 generadores por stubs que registran sus llamadas."""
    calls = []

    def make(name):
        def stub(model_id, metadata, out_dir, device="cuda"):
            calls.append({
                "fn": name, "model_id": model_id,
                "out": Path(out_dir).name, "device": device,
            })
        return stub

    monkeypatch.setattr(ga, "generate_musicgen", make("musicgen"))
    monkeypatch.setattr(ga, "generate_audioldm2", make("audioldm2"))
    monkeypatch.setattr(ga, "generate_stable_audio", make("stable"))
    return calls


def _require_metadata():
    if not (ROOT / "testset_metadata.csv").exists():
        pytest.skip("testset_metadata.csv no presente")


def test_main_routes_selected_models(tmp_path, monkeypatch, stub_generators):
    _require_metadata()
    monkeypatch.setattr(sys, "argv", [
        "generate_all.py", "--models", "musicgen-small", "audioldm2",
        "--device", "cpu", "--wavs-dir", str(tmp_path),
    ])
    ga.main()
    assert {c["fn"] for c in stub_generators} == {"musicgen", "audioldm2"}
    model_ids = {c["model_id"] for c in stub_generators}
    assert "facebook/musicgen-small" in model_ids
    assert "cvssp/audioldm2" in model_ids
    assert (tmp_path / "musicgen-small").is_dir()
    assert (tmp_path / "audioldm2").is_dir()


def test_main_default_models(tmp_path, monkeypatch, stub_generators):
    _require_metadata()
    monkeypatch.setattr(sys, "argv", [
        "generate_all.py", "--device", "cpu", "--wavs-dir", str(tmp_path),
    ])
    ga.main()
    assert sorted(c["out"] for c in stub_generators) == ["musicgen-medium", "musicgen-small"]


def test_main_all_four_models(tmp_path, monkeypatch, stub_generators):
    _require_metadata()
    monkeypatch.setattr(sys, "argv", [
        "generate_all.py", "--models",
        "musicgen-small", "musicgen-medium", "audioldm2", "stable-audio-open",
        "--device", "cpu", "--wavs-dir", str(tmp_path),
    ])
    ga.main()
    assert len(stub_generators) == 4
    assert {c["model_id"] for c in stub_generators} == {
        "facebook/musicgen-small", "facebook/musicgen-medium",
        "cvssp/audioldm2", "stabilityai/stable-audio-open-1.0",
    }


def test_main_passes_device_to_generators(tmp_path, monkeypatch, stub_generators):
    _require_metadata()
    monkeypatch.setattr(sys, "argv", [
        "generate_all.py", "--models", "musicgen-small",
        "--device", "cpu", "--wavs-dir", str(tmp_path),
    ])
    ga.main()
    assert stub_generators[0]["device"] == "cpu"


def test_main_rejects_invalid_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_all.py", "--models", "bogus-model"])
    with pytest.raises(SystemExit):  # argparse rechaza un choice inválido
        ga.main()
