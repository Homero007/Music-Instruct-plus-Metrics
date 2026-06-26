"""
model.py — Reward model (MLP) + pérdida pairwise (Bradley-Terry).

POR QUÉ PAIRWISE Y NO REGRESIÓN ESCALAR:
  La "calidad musical" es ordinal y subjetiva: no existe un número absoluto
  "esto vale 0.73". Sí existen comparaciones consistentes ("A suena mejor que
  B"). Bradley-Terry parametriza la probabilidad
       P(A > B) = sigmoid(r(A) - r(B))
  y el modelo aprende SOLO diferencias relativas. Esto es exactamente lo que
  hacen los reward models de RLHF.

  Ventaja para nosotros: para arrancar SIN labels humanos, podemos generar
  pares automáticos donde "A" es un clip real del dataset y "B" es una
  candidata generada. El modelo aprende a distinguir música real vs. sintética,
  lo cual es un proxy débil pero útil para el re-ranking.

Requiere torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RewardModelConfig:
    in_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    activation: str = "gelu"   # "gelu" | "relu"


class RewardMLP(nn.Module):
    """MLP que toma un vector de features y devuelve un score escalar."""

    def __init__(self, cfg: RewardModelConfig):
        super().__init__()
        self.cfg = cfg
        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU}[cfg.activation]
        layers: list[nn.Module] = []
        prev = cfg.in_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), act_cls(), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_dim) → (B,)
        return self.net(x).squeeze(-1)


# ── Pérdidas y métricas ──────────────────────────────────────────────────────

def bradley_terry_loss(
    score_pref: torch.Tensor,
    score_rej: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Pérdida pairwise: -log sigmoid(r_pref - r_rej - margin).

    `margin` introduce un colchón mínimo (hinge-soft). 0 = Bradley-Terry puro.
    """
    return -F.logsigmoid(score_pref - score_rej - margin).mean()


def pairwise_accuracy(score_pref: torch.Tensor, score_rej: torch.Tensor) -> float:
    """Fracción de pares donde el preferido obtuvo score más alto."""
    return float((score_pref > score_rej).float().mean().item())


# ── Persistencia ─────────────────────────────────────────────────────────────

def save_model(model: RewardMLP, path: str) -> None:
    """Guarda config + state_dict para reconstruir sin código duplicado."""
    torch.save(
        {
            "config": {
                "in_dim": model.cfg.in_dim,
                "hidden_dims": list(model.cfg.hidden_dims),
                "dropout": model.cfg.dropout,
                "activation": model.cfg.activation,
            },
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: str, device: str = "cpu") -> RewardMLP:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = RewardModelConfig(
        in_dim=int(blob["config"]["in_dim"]),
        hidden_dims=tuple(blob["config"]["hidden_dims"]),
        dropout=float(blob["config"]["dropout"]),
        activation=str(blob["config"]["activation"]),
    )
    model = RewardMLP(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    model.to(device)
    return model
