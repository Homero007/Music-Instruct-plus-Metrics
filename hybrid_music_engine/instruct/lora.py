"""
lora.py — Adaptadores LoRA de bajo rango + utilidades de congelación.

Punto 3 de la estrategia (Instruct-MusicGen): el transformador base (MusicGen) y
el codificador T5 ya saben sintetizar y estructurar música. Para enseñarles a
"obedecer comandos de edición" sin arruinar esa capacidad:

  1. Se congela TODO el modelo base y T5.
  2. Se inyectan adaptadores de bajo rango A·B SOLO en las proyecciones de las
     capas de atención cruzada (Q, K, V) que conectan texto/audio con los tokens
     musicales.

LoRA reemplaza el cómputo  y = W x  por  y = W x + (alpha/r) · B(A x), donde W
queda congelado y solo A (r×in) y B (out×r) se entrenan. B se inicializa a cero,
de modo que al inicio del fine-tuning el modelo es IDÉNTICO al preentrenado y la
edición se aprende de forma incremental sin colapsar el conocimiento previo.

Requiere: torch.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Envuelve un nn.Linear existente con un adaptador de bajo rango.

    El peso original `base` se congela; solo `lora_A` y `lora_B` se entrenan.
    Salida: base(x) + scaling · dropout(x) @ Aᵀ @ Bᵀ
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("El rango r debe ser > 0")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        # Congelar el peso base original
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # A ~ kaiming, B = 0  →  delta inicial = 0
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + self.scaling * delta

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}"


def wrap_linear(linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0) -> LoRALinear:
    """Envuelve un nn.Linear en un LoRALinear (idempotente: no re-envuelve)."""
    if isinstance(linear, LoRALinear):
        return linear
    return LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)


def inject_lora(
    module: nn.Module,
    target_names: tuple[str, ...] = ("q_proj", "k_proj", "v_proj"),
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> int:
    """
    Recorre `module` y envuelve en LoRA todo nn.Linear cuyo nombre de atributo
    coincida con `target_names`. Por defecto solo Q, K, V (las proyecciones de
    atención cruzada). Devuelve cuántos linears fueron envueltos.

    Para limitar a las capas de cross-attention, pásale directamente esos
    submódulos (p. ej. los AudioFusionModule y las text-cross-attentions), no
    el modelo entero.
    """
    count = 0
    for parent in module.modules():
        for attr_name, child in list(parent.named_children()):
            if attr_name in target_names and isinstance(child, nn.Linear):
                setattr(parent, attr_name, wrap_linear(child, r=r, alpha=alpha, dropout=dropout))
                count += 1
    return count


def freeze_all(module: nn.Module) -> None:
    """Congela todos los parámetros de un módulo."""
    for param in module.parameters():
        param.requires_grad = False


def _is_gate(name: str) -> bool:
    """True si el parámetro es una compuerta de fusión (AudioFusionModule.gate)."""
    return name == "gate" or name.endswith(".gate")


def mark_only_lora_trainable(
    module: nn.Module,
    train_bias: bool = False,
    train_gates: bool = True,
) -> None:
    """
    Congela todo excepto los parámetros del ADAPTADOR. El adaptador son:
      • los pesos LoRA (lora_A / lora_B), y
      • por defecto, las compuertas de fusión (`gate`).

    La compuerta DEBE entrenarse: arranca en 0 (fusión = identidad) y su gradiente
    inicial es no nulo, por lo que se "abre" en el primer paso y deja que los LoRA
    del ramal de audio empiecen a recibir señal. Si la congelaras, el ramal de
    fusión de audio nunca aprendería. Este es el estado correcto para fine-tuning
    de edición: base + T5 congelados, solo el adaptador aprende.
    """
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif train_gates and _is_gate(name):
            param.requires_grad = True
        elif train_bias and name.endswith(".bias"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Itera solo los parámetros LoRA (lora_A / lora_B)."""
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def adapter_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Itera los parámetros del adaptador (LoRA + gates) para el optimizador."""
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name or _is_gate(name):
            yield param


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Cuenta parámetros totales, entrenables, LoRA y de compuerta."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    lora = sum(p.numel() for p in lora_parameters(module))
    gates = sum(p.numel() for name, p in module.named_parameters() if _is_gate(name))
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "gates": gates,
        "adapter": lora + gates,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }
