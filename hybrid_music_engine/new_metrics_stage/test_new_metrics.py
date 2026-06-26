"""
Verificación end-to-end de la etapa new_metrics.

Genera, en formato real:
  • Audio por género (real/ y generado/<genero>/) con tempo y timbre distintos.
  • t5_seq/genre__*.npz para el t-SNE de palabras.
  • Probabilidades para KLD.

Verifica que la etapa produce gráficos (FAD/tempo, t-SNE palabras, KLD) y un
report.json, SIN tocar el reward model.

Ejecutar:  python test_new_metrics.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import soundfile as sf

from new_metrics.pipeline import run_new_metrics

GENRES = ("classical", "electronic", "reggaeton")


def _tone(path: Path, bpm: float, tone_hz: float, sr=22050, seconds=5.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    y = 0.05 * np.sin(2 * np.pi * tone_hz * t)
    period = 60.0 / bpm
    for k in range(int(seconds / period)):
        idx = int(k * period * sr)
        if idx < n:
            env = np.exp(-np.arange(min(800, n - idx)) / 60.0)
            y[idx:idx + len(env)] += env * 0.6
    if noise:
        y += noise * rng.standard_normal(n)
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr)


def _write_t5(t5_dir: Path, seed=0):
    rng = np.random.default_rng(seed)
    centers = {g: rng.normal(3 * i, 0.3, 768) for i, g in enumerate(GENRES)}
    for g in GENRES:
        for t in range(2):
            L = 18
            hidden = (centers[g] + rng.normal(0, 0.4, (L, 768))).astype(np.float16)
            p = t5_dir / f"genre__{g}_{t}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, hidden=hidden, text=np.array(f"{g}"),
                                shape=np.array(hidden.shape))


def test_new_metrics_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        real = root / "real"
        gen = root / "generated"
        params = {"classical": (72, 220), "electronic": (128, 440), "reggaeton": (95, 660)}
        for g, (bpm, hz) in params.items():
            for i in range(3):
                _tone(real / g / f"r_{i}.wav", bpm, hz, seed=i)
                _tone(gen / g / f"c_{i}.wav", bpm + 2, hz * 1.05, noise=0.05, seed=100 + i)

        t5 = root / "encodings_v2" / "t5_seq"
        _write_t5(t5)

        # Probabilidades para KLD (3 clases).
        real_probs = root / "real_probs.npy"
        gen_probs = root / "gen_probs.npy"
        np.save(real_probs, np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]))
        np.save(gen_probs, np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]))

        report = run_new_metrics(
            generated_root=gen,
            out_dir=root / "out",
            real_root=real,
            t5_dir=t5,
            metrics=["fad", "tempos_std"],
            fad_extractor="mel",
            kld_real_probs=real_probs,
            kld_gen_probs=gen_probs,
            knn=6,
        )

        # Report + carpeta de gráficos.
        assert (root / "out" / "report.json").exists()
        plots = report["plots"]
        assert len(plots) >= 3, f"esperaba varios gráficos, hubo {len(plots)}"
        # Hay gráficos de audio, de t-SNE de palabras y de KLD.
        names = " ".join(Path(p).name for p in plots)
        assert "audio_" in names
        assert "tsne_tsne_palabras_3generos" in names or "palabras_3generos" in names
        assert "kld_distribuciones" in names
        # KLD numérico calculado.
        assert isinstance(report["kld"].get("kld"), float)
        # No se escribió nada de reward model / ranking.
        assert not list((root / "out").rglob("ranking*.json"))


def test_runs_without_real_only_tsne():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        gen = root / "generated"
        _tone(gen / "classical" / "a.wav", 72, 220, seed=1)
        t5 = root / "t5_seq"
        _write_t5(t5)
        # Sin --real: FAD se omite; t-SNE de palabras igual produce gráficos.
        report = run_new_metrics(
            generated_root=gen, out_dir=root / "out", t5_dir=t5,
            metrics=["fad"], fad_extractor="mel",
        )
        names = " ".join(Path(p).name for p in report["plots"])
        assert "palabras_3generos" in names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("TODAS LAS PRUEBAS DE NEW_METRICS PASARON")
