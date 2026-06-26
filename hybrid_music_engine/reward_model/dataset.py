"""
dataset.py — Pares de preferencia para entrenar el reward model.

Soporta dos fuentes:

1) MANIFIESTO EXPLÍCITO (JSONL o CSV)
   Cada línea es un par (preferred, rejected). El valor de cada lado puede ser:
     • una ruta a un .mid (se computan features con metrics_fn), o
     • una ruta a un .json con métricas ya computadas, o
     • un dict de métricas inline.

   JSONL ejemplo:
     {"preferred": "data/real/jazz_01.mid", "rejected": "data/ranked/X/candidate-03/generated.mid"}
     {"preferred": "data/ranked/Y/candidate-02/metrics.json", "rejected": {"tempo": 120, ...}}

2) BOOTSTRAP AUTOMÁTICO (sin labels humanos)
   `build_bootstrap_pairs(real_dir, generated_dir, ...)` empareja muestras del
   dataset real (Jamendo procesado) contra candidatas generadas, asignando
   "preferred = real" siempre. El modelo aprende a empujar las generaciones
   hacia la distribución de música real — sesgo conservador pero útil como
   punto de partida.

Las pares humanas siempre dominan: si tienes ambos, mezcla y prioriza humanas.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

log = logging.getLogger(__name__)

MetricsFn = Callable[[Path], Mapping[str, Any]]


@dataclass
class PreferencePair:
    preferred: Mapping[str, Any]
    rejected: Mapping[str, Any]
    weight: float = 1.0
    source: str = ""    # trazabilidad: "human" | "bootstrap_real_vs_gen" | ...


# ── Resolución flexible de "lo que sea" → métricas ────────────────────────────

def _resolve_side(
    side: Any,
    metrics_fn: MetricsFn | None,
    cache: dict[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """
    Convierte una entrada heterogénea en un dict de métricas.

    Acepta:
      • dict directo
      • ruta a .json con métricas
      • ruta a .mid → llama metrics_fn (si está disponible)
    """
    if isinstance(side, Mapping):
        return dict(side)
    if isinstance(side, (str, Path)):
        s = str(side)
        if s in cache:
            return cache[s]
        path = Path(s)
        if not path.exists():
            log.warning("Ruta no existe: %s", path)
            return None
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("No se pudo leer %s: %s", path, exc)
                return None
            cache[s] = data
            return data
        if path.suffix.lower() in {".mid", ".midi"}:
            if metrics_fn is None:
                log.warning("MIDI sin metrics_fn: %s", path)
                return None
            try:
                data = metrics_fn(path)
            except Exception as exc:    # noqa: BLE001 — registramos y seguimos
                log.warning("metrics_fn falló en %s: %s", path, exc)
                return None
            cache[s] = data
            return data
    log.warning("Tipo no reconocido para par: %r", type(side))
    return None


# ── Lectura de manifiestos ────────────────────────────────────────────────────

def read_preference_manifest(
    path: Path,
    metrics_fn: MetricsFn | None = None,
) -> list[PreferencePair]:
    """Lee JSONL o CSV de pares de preferencia."""
    cache: dict[str, Mapping[str, Any]] = {}
    rows = _read_rows(path)
    pairs: list[PreferencePair] = []
    for row in rows:
        pref = _resolve_side(row.get("preferred"), metrics_fn, cache)
        rej = _resolve_side(row.get("rejected"), metrics_fn, cache)
        if pref is None or rej is None:
            continue
        try:
            weight = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        pairs.append(PreferencePair(pref, rej, weight=weight, source=str(row.get("source", "human"))))
    return pairs


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        out: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(r) for r in data]
        return [dict(r) for r in data.get("pairs", [])]
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    raise ValueError(f"Extensión no soportada para pares: {suffix}")


# ── Bootstrap: pares automáticos real vs. generado ────────────────────────────

def _gather_midi(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".mid", ".midi"})


def build_bootstrap_pairs(
    real_dir: Path,
    generated_dir: Path,
    metrics_fn: MetricsFn,
    max_pairs: int = 2000,
    seed: int = 42,
) -> list[PreferencePair]:
    """
    Empareja MIDIs reales contra MIDIs generados. preferred = real.

    LIMITACIÓN ASUMIDA: el reward aprendido será un detector de "música real
    vs. sintética", no un juez de calidad fino. Sirve como bootstrap; lo ideal
    es complementar con pares humanos después.
    """
    rng = random.Random(seed)
    reals = _gather_midi(real_dir)
    gens = _gather_midi(generated_dir)
    if not reals or not gens:
        log.warning("Bootstrap vacío: reales=%d, generados=%d", len(reals), len(gens))
        return []

    n = min(max_pairs, len(reals) * len(gens))
    pairs: list[PreferencePair] = []
    cache: dict[str, Mapping[str, Any]] = {}
    for _ in range(n):
        r = rng.choice(reals)
        g = rng.choice(gens)
        pref = _resolve_side(r, metrics_fn, cache)
        rej = _resolve_side(g, metrics_fn, cache)
        if pref is None or rej is None:
            continue
        pairs.append(PreferencePair(pref, rej, weight=1.0, source="bootstrap_real_vs_gen"))
    log.info("Bootstrap: %d pares construidos (real=%d, gen=%d)", len(pairs), len(reals), len(gens))
    return pairs


# ── Iterador con mezcla de fuentes ────────────────────────────────────────────

def merge_pairs(
    human_pairs: Iterable[PreferencePair],
    bootstrap_pairs: Iterable[PreferencePair],
    human_weight_multiplier: float = 3.0,
) -> list[PreferencePair]:
    """
    Combina ambas fuentes, multiplicando el peso de las humanas para que
    dominen el gradiente. Si no hay humanas, devuelve solo bootstrap.
    """
    merged: list[PreferencePair] = []
    for p in human_pairs:
        merged.append(PreferencePair(p.preferred, p.rejected,
                                     weight=p.weight * human_weight_multiplier,
                                     source=p.source or "human"))
    for p in bootstrap_pairs:
        merged.append(p)
    return merged
