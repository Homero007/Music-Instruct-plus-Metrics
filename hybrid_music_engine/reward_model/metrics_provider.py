"""
metrics_provider.py — Cómo obtener métricas para un MIDI.

Dos rutas, intentadas en orden:

  1. API LOCAL del proyecto: si `HYBRID_ENGINE_API_URL` apunta a un servidor
     corriendo, hacemos POST a /api/metrics/midi (lo que ya usa tu pipeline).
     Esto garantiza que las métricas son EXACTAMENTE las que produce el resto
     del sistema.

  2. FALLBACK LOCAL: si la API no responde, calculamos un subconjunto razonable
     con `pretty_midi`. Es estable y portable, pero puede no incluir todas las
     métricas exóticas del backend (swing, syncopation custom, etc.).

El operador puede forzar el modo con --metrics-mode {api,local,auto}.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

DEFAULT_API = os.environ.get("HYBRID_ENGINE_API_URL", "http://127.0.0.1:8100")


# ── Vía API ──────────────────────────────────────────────────────────────────

def metrics_via_api(midi_path: Path, api_url: str = DEFAULT_API, timeout: float = 30) -> Mapping[str, Any]:
    body = json.dumps({"midi_path": str(midi_path)}).encode("utf-8")
    req = Request(
        f"{api_url.rstrip('/')}/api/metrics/midi",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload  # se pasa tal cual; flatten_metrics resuelve {"metrics": {...}}


# ── Fallback local con pretty_midi ────────────────────────────────────────────

def metrics_local(midi_path: Path) -> Mapping[str, Any]:
    """Métricas mínimas sin tocar el backend; útil para tests y entornos sueltos."""
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError(
            "pretty_midi no instalado. `pip install pretty_midi` o usa --metrics-mode api."
        ) from exc

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    duration = float(pm.get_end_time())
    n_notes = len(notes)

    if n_notes == 0:
        return {
            "duration_seconds": duration, "num_notes": 0, "tempo_bpm": 0,
            "note_density": 0, "mean_pitch": 0, "pitch_range": 0,
            "mean_velocity": 0, "velocity_std": 0, "mean_duration": 0,
            "pitch_classes": [0] * 12,
        }

    import numpy as np
    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    vels = np.array([n.velocity for n in notes], dtype=np.float64)
    durs = np.array([n.end - n.start for n in notes], dtype=np.float64)
    pcs = np.zeros(12, dtype=np.float64)
    for p in pitches:
        pcs[int(p) % 12] += 1
    tempi = pm.get_tempo_changes()[1]
    tempo = float(tempi.mean()) if len(tempi) else 0.0

    return {
        "duration_seconds": duration,
        "num_notes": n_notes,
        "tempo_bpm": tempo,
        "note_density": n_notes / duration if duration > 0 else 0.0,
        "mean_pitch": float(pitches.mean()),
        "pitch_range": float(pitches.max() - pitches.min()),
        "mean_velocity": float(vels.mean()),
        "velocity_std": float(vels.std()),
        "mean_duration": float(durs.mean()),
        "polyphony": n_notes / duration if duration > 0 else 0.0,  # aproximado
        "num_layers": len(pm.instruments),
        "pitch_classes": pcs.tolist(),
    }


# ── Factoría auto ────────────────────────────────────────────────────────────

def make_metrics_fn(mode: str = "auto", api_url: str = DEFAULT_API):
    """
    Devuelve una `metrics_fn(path) -> dict`.

    mode = "api"   → solo API; falla si no responde
    mode = "local" → solo fallback local
    mode = "auto"  → intenta API; si falla, cae a local. Decide la primera vez
                     y mantiene la decisión para no spamear conexiones.
    """
    mode = mode.lower()
    if mode == "api":
        return lambda p: metrics_via_api(p, api_url=api_url)
    if mode == "local":
        return metrics_local
    if mode != "auto":
        raise ValueError(f"mode desconocido: {mode}")

    state = {"resolved": None}

    def fn(path: Path):
        if state["resolved"] == "local":
            return metrics_local(path)
        if state["resolved"] == "api":
            return metrics_via_api(path, api_url=api_url)
        # Primera llamada: probamos API
        try:
            out = metrics_via_api(path, api_url=api_url)
            state["resolved"] = "api"
            log.info("metrics provider: usando API en %s", api_url)
            return out
        except (URLError, OSError, TimeoutError) as exc:
            log.info("API no disponible (%s); usando fallback local con pretty_midi.", exc)
            state["resolved"] = "local"
            return metrics_local(path)

    return fn
