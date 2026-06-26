"""Pruebas del modelo y el entrenamiento (train_colab.py).

Cubre: dataset sintético, forward del Transformer, un paso de entrenamiento,
guardado/recarga de checkpoint y un smoke test de la CLI completa.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
import numpy as np

import train_colab as tc


def test_token_dataset_shape_and_range():
    ds = tc.TokenDataset(num_samples=10, seq_len=16, vocab_size=64, seed=0)
    assert len(ds) == 10
    item = ds[0]
    assert tuple(item.shape) == (16,)
    assert item.dtype == torch.long
    assert int(item.min()) >= 0 and int(item.max()) < 64


def test_token_dataset_deterministic():
    # El dataset siembra el RNG global en el constructor, así que el contrato
    # de reproducibilidad es: construir y leer el primer ítem da el mismo tensor.
    a = tc.TokenDataset(num_samples=4, seq_len=8, vocab_size=32, seed=123)
    first_a = a[0].clone()
    b = tc.TokenDataset(num_samples=4, seq_len=8, vocab_size=32, seed=123)
    first_b = b[0].clone()
    assert torch.equal(first_a, first_b)


def test_transformer_forward_shape():
    model = tc.SimpleTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=1, seq_len=16)
    x = torch.randint(0, 64, (3, 16))
    out = model(x)
    assert tuple(out.shape) == (3, 16, 64)
    assert torch.isfinite(out).all()


def test_train_epoch_updates_params_and_finite_loss():
    torch.manual_seed(0)
    ds = tc.TokenDataset(num_samples=8, seq_len=16, vocab_size=64, seed=0)
    dl = torch.utils.data.DataLoader(ds, batch_size=4)
    model = tc.SimpleTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=1, seq_len=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    before = model.head.weight.detach().clone()
    loss = tc.train_epoch(model, dl, opt, loss_fn, torch.device("cpu"))

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    after = model.head.weight.detach()
    assert not torch.allclose(before, after), "los parámetros deberían actualizarse tras un paso"


def test_save_and_load_checkpoint(tmp_path):
    model = tc.SimpleTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=1, seq_len=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "ckpt.pt"

    assert tc.save_checkpoint(model, opt, epoch=3, loss=1.23, path=ckpt) is True
    assert ckpt.exists()

    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("epoch", "model_state_dict", "optimizer_state_dict", "loss", "timestamp"):
        assert key in data
    assert data["epoch"] == 3

    # El modelo recargado produce la misma salida (en eval, sin dropout).
    model.eval()
    x = torch.randint(0, 64, (2, 16))
    out_before = model(x)
    model2 = tc.SimpleTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=1, seq_len=16)
    model2.load_state_dict(data["model_state_dict"])
    model2.eval()
    out_after = model2(x)
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_cli_smoke(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    res_dir = tmp_path / "res"
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "train_colab.py"),
            "--epochs", "2", "--num-samples", "8", "--batch-size", "4", "--save-every", "1",
            "--checkpoint-dir", str(ckpt_dir), "--results-dir", str(res_dir),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    summary = json.loads((res_dir / "summary.json").read_text())
    assert summary["total_epochs"] == 2
    assert "final_loss" in summary and "best_loss" in summary
    assert (res_dir / "training_metrics.csv").exists()
    assert (res_dir / "training_metrics.json").exists()
    assert list(ckpt_dir.glob("model_ep*.pt")), "se esperaba al menos un checkpoint"


# ── Robustez de reproducibilidad del dataset ────────────────────────────────

def test_dataset_order_independent():
    # Con RNG por índice, acceder en cualquier orden da el mismo tensor.
    ds = tc.TokenDataset(num_samples=5, seq_len=8, vocab_size=32, seed=7)
    a2 = ds[2].clone()
    _ = ds[0]
    _ = ds[4]
    assert torch.equal(ds[2], a2)
    # Dos datasets con la misma semilla coinciden en todos los índices.
    ds2 = tc.TokenDataset(num_samples=5, seq_len=8, vocab_size=32, seed=7)
    for i in range(5):
        assert torch.equal(ds[i], ds2[i])


def test_dataset_different_seed_differs():
    a = tc.TokenDataset(num_samples=2, seq_len=8, vocab_size=64, seed=1)
    b = tc.TokenDataset(num_samples=2, seq_len=8, vocab_size=64, seed=2)
    assert not torch.equal(a[0], b[0])


def test_dataset_does_not_touch_global_rng():
    # Construir/usar el dataset NO debe alterar el RNG global de numpy.
    np.random.seed(999)
    expected = np.random.rand()
    np.random.seed(999)
    ds = tc.TokenDataset(num_samples=3, seq_len=8, vocab_size=32, seed=0)
    _ = ds[0]
    _ = ds[1]
    assert np.random.rand() == expected


# ── Modelo: casos adicionales ───────────────────────────────────────────────

def test_forward_shorter_seq_len():
    model = tc.SimpleTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=1, seq_len=32)
    x = torch.randint(0, 64, (2, 10))  # seq menor que seq_len configurado
    out = model(x)
    assert tuple(out.shape) == (2, 10, 64)


def test_gradients_flow():
    model = tc.SimpleTransformer(vocab_size=64, d_model=32, nhead=2, num_layers=1, seq_len=16)
    x = torch.randint(0, 64, (2, 16))
    model(x).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "debería haber gradientes"
    assert any(g.abs().sum().item() > 0 for g in grads), "algún gradiente debe ser no nulo"


# ── CLI: reproducibilidad y frecuencia de checkpoints ───────────────────────

def test_cli_reproducible_same_seed(tmp_path):
    def run(tag):
        d = tmp_path / tag
        proc = subprocess.run(
            [
                sys.executable, str(ROOT / "train_colab.py"),
                "--epochs", "3", "--num-samples", "8", "--batch-size", "4",
                "--save-every", "10", "--seed", "123",
                "--checkpoint-dir", str(d / "c"), "--results-dir", str(d / "r"),
            ],
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads((d / "r" / "summary.json").read_text())["final_loss"]

    assert run("a") == pytest.approx(run("b"), rel=1e-4)


def test_cli_save_every_frequency(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "train_colab.py"),
            "--epochs", "4", "--num-samples", "8", "--batch-size", "4", "--save-every", "2",
            "--checkpoint-dir", str(tmp_path / "c"), "--results-dir", str(tmp_path / "r"),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    ckpts = sorted((tmp_path / "c").glob("model_ep*.pt"))
    assert len(ckpts) == 2  # epochs 2 y 4
