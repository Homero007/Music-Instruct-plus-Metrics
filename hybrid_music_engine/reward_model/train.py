"""
train.py — Entrenamiento del reward model con pérdida pairwise.

Pipeline:
  pares de preferencia → ajustar FeatureSchema → vectorizar →
  shuffle/split → entrenamiento con Bradley-Terry → early stopping → guardar.

Requiere torch.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import PreferencePair
from .features import FeatureSchema
from .model import RewardMLP, RewardModelConfig, bradley_terry_loss, pairwise_accuracy, save_model

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    patience: int = 5            # early stop si val_acc no mejora N epochs
    margin: float = 0.0
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    seed: int = 42
    device: str = "cpu"


class PairDataset(Dataset):
    """(pref_vec, rej_vec, weight) listos para el optimizador."""

    def __init__(self, pairs: list[PreferencePair], schema: FeatureSchema, standardize: bool = True):
        self.schema = schema
        self.pref = schema.vectorize_batch(p.preferred for p in pairs)
        self.rej = schema.vectorize_batch(p.rejected for p in pairs)
        if standardize:
            self.pref = schema.standardize(self.pref)
            self.rej = schema.standardize(self.rej)
        self.weights = np.asarray([p.weight for p in pairs], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.weights)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        return self.pref[idx], self.rej[idx], float(self.weights[idx])


def _collate(batch):
    pref = torch.from_numpy(np.stack([b[0] for b in batch]))
    rej = torch.from_numpy(np.stack([b[1] for b in batch]))
    w = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return pref, rej, w


def split_pairs(
    pairs: list[PreferencePair], val_fraction: float, seed: int
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def train_reward_model(
    pairs: list[PreferencePair],
    out_dir: Path,
    schema: FeatureSchema | None = None,
    cfg: TrainConfig | None = None,
) -> dict:
    """
    Entrena un reward model y lo guarda en out_dir.

    Devuelve un dict con métricas finales y rutas a artefactos.
    """
    if not pairs:
        raise ValueError("Sin pares para entrenar.")
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # 1) Esquema (si no se pasa, se ajusta sobre TODOS los lados)
    if schema is None:
        all_sides: list = [p.preferred for p in pairs] + [p.rejected for p in pairs]
        schema = FeatureSchema.fit(all_sides)
        log.info("Esquema ajustado: dim=%d", schema.dim)
    schema_path = out_dir / "schema.json"
    schema.to_json(schema_path)

    # 2) Split + DataLoaders
    train_pairs, val_pairs = split_pairs(pairs, cfg.val_fraction, cfg.seed)
    train_ds = PairDataset(train_pairs, schema)
    val_ds = PairDataset(val_pairs, schema) if val_pairs else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)
        if val_ds is not None and len(val_ds) > 0 else None
    )

    # 3) Modelo + optimizador
    model_cfg = RewardModelConfig(in_dim=schema.dim, hidden_dims=cfg.hidden_dims, dropout=cfg.dropout)
    model = RewardMLP(model_cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_accs: list[float] = []
        for pref, rej, w in train_loader:
            pref = pref.to(cfg.device); rej = rej.to(cfg.device); w = w.to(cfg.device)
            sp = model(pref); sr = model(rej)
            # Pérdida ponderada: -log sigmoid(sp - sr - margin), pesada por w
            per = -torch.nn.functional.logsigmoid(sp - sr - cfg.margin)
            loss = (per * w).sum() / w.sum().clamp(min=1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))
            train_accs.append(pairwise_accuracy(sp, sr))

        # Validación
        val_loss = float("nan"); val_acc = float("nan")
        if val_loader is not None:
            model.eval()
            losses: list[float] = []
            accs: list[float] = []
            with torch.no_grad():
                for pref, rej, _w in val_loader:
                    pref = pref.to(cfg.device); rej = rej.to(cfg.device)
                    sp = model(pref); sr = model(rej)
                    losses.append(float(bradley_terry_loss(sp, sr, cfg.margin).item()))
                    accs.append(pairwise_accuracy(sp, sr))
            val_loss = float(np.mean(losses)) if losses else float("nan")
            val_acc = float(np.mean(accs)) if accs else float("nan")

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_acc": float(np.mean(train_accs)),
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        log.info(
            "epoch %02d | train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f",
            epoch, history[-1]["train_loss"], history[-1]["train_acc"], val_loss, val_acc,
        )

        # Early stopping con respecto a val_acc (o train_acc si no hay val)
        ref_acc = val_acc if val_loader is not None and not np.isnan(val_acc) else history[-1]["train_acc"]
        if ref_acc > best_val_acc + 1e-4:
            best_val_acc = ref_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                log.info("Early stopping en epoch %d (mejor val_acc=%.3f)", epoch, best_val_acc)
                break

    # Restaurar mejor estado y guardar
    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = out_dir / "reward_model.pt"
    save_model(model, str(model_path))

    history_path = out_dir / "history.json"
    import json
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "history_path": str(history_path),
        "best_val_acc": best_val_acc,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "feature_dim": schema.dim,
    }


def evaluate(
    model: RewardMLP,
    pairs: Iterable[PreferencePair],
    schema: FeatureSchema,
    device: str = "cpu",
) -> dict:
    """Evalúa accuracy pairwise sobre un conjunto."""
    ds = PairDataset(list(pairs), schema)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=_collate)
    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for pref, rej, _w in loader:
            pref = pref.to(device); rej = rej.to(device)
            sp = model(pref); sr = model(rej)
            accs.append(pairwise_accuracy(sp, sr))
    return {"pairwise_accuracy": float(np.mean(accs)) if accs else float("nan"), "n_pairs": len(ds)}
