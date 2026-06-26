"""
features.py — Normalización de features musicales a vector fijo.

El pipeline existente (`/api/metrics/midi`) devuelve `{"metrics": {...}}` con
campos como tempo, densidad, syncopation, pitch classes, etc. Distintos puntos
del código pueden usar nombres distintos para lo mismo (tempo / tempo_bpm /
estimated_tempo), así que este módulo expone un FeatureSchema tolerante.

Reglas:
  • Cada campo es OPCIONAL: si falta, se imputa con la mediana del conjunto de
    entrenamiento (o 0 al inicio). Una bandera `<name>__missing` (0/1) se añade
    al vector para que el modelo pueda aprender a desconfiar de imputaciones.
  • Listas de longitud variable (pitch classes, histogramas) se compactan a
    estadísticos estables (mean, std, entropía, top-1).
  • La salida es un vector float32 de tamaño FIJO y un dict trazable. NUNCA se
    cambia el orden de los campos: persistir/cargar el schema con `to_json` /
    `from_json` para reproducir el vector exacto en inferencia.

Sin dependencias externas más allá de numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# ── Aliases: nombre canónico → posibles claves observadas en el wild ──────────
#
# Mantén el nombre canónico en la izquierda. Para añadir una variante nueva,
# basta con incluirla en la lista de aliases SIN cambiar el orden de campos.

SCALAR_ALIASES: dict[str, tuple[str, ...]] = {
    "tempo_bpm":         ("tempo_bpm", "tempo", "estimated_tempo", "bpm"),
    "note_density":      ("note_density", "density", "notes_per_second", "events_per_second"),
    "polyphony":         ("polyphony", "avg_polyphony", "mean_polyphony"),
    "pitch_range":       ("pitch_range", "range", "pitch_span"),
    "mean_pitch":        ("mean_pitch", "avg_pitch", "pitch_mean"),
    "mean_velocity":     ("mean_velocity", "avg_velocity", "velocity_mean"),
    "velocity_std":      ("velocity_std", "velocity_stdev"),
    "mean_duration":     ("mean_duration", "avg_duration", "duration_mean"),
    "duration_seconds":  ("duration_seconds", "duration", "length_seconds"),
    "swing":             ("swing", "swing_ratio", "swing_amount"),
    "syncopation":       ("syncopation", "syncopation_score"),
    "energy":            ("energy", "rms_energy"),
    "rest_ratio":        ("rest_ratio", "silence_ratio"),
    "num_layers":        ("num_layers", "n_layers", "num_tracks"),
    "num_notes":         ("num_notes", "n_notes", "note_count"),
}

# Vectores de tamaño variable que se compactan a estadísticos fijos.
VECTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "pitch_classes":     ("pitch_classes", "pitch_class_histogram", "pcp"),
    "interval_histogram": ("interval_histogram", "intervals_hist"),
}

# Estadísticos derivados de cada vector (mismo orden = mismas columnas).
VECTOR_STATS = ("mean", "std", "max", "entropy_norm")


def _first_match(d: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):                      # bool es subclase de int
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        # Histograma como dict: {pitch_class_str: count}
        try:
            arr = np.asarray([float(v) for v in value.values()], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
    else:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _vector_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(arr.mean())
    std = float(arr.std())
    mx = float(arr.max())
    # Entropía normalizada en [0, 1] (uniforme = 1).
    if arr.sum() <= 0 or arr.size <= 1:
        ent = 0.0
    else:
        p = arr / arr.sum()
        p = p[p > 0]
        ent = float(-np.sum(p * np.log(p)) / math.log(arr.size))
    return mean, std, mx, ent


# Set de TODOS los aliases conocidos: no aplanamos sub-dicts cuya clave coincida.
_KNOWN_ALIASES: frozenset[str] = frozenset(
    a for aliases in (*SCALAR_ALIASES.values(), *VECTOR_ALIASES.values()) for a in aliases
)


def flatten_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Acepta tanto `{"metrics": {...}}` como `{...}` directamente, y también
    aplana un nivel de anidación tipo namespace (`{"global": {...}, "by_layer":
    {...}}`). NO aplana sub-dicts cuya clave sea un nombre conocido (como
    `pitch_classes`), porque esos sub-dicts SON el valor (histograma).
    """
    if isinstance(metrics, Mapping) and "metrics" in metrics and isinstance(metrics["metrics"], Mapping):
        metrics = metrics["metrics"]
    flat: dict[str, Any] = {}
    for k, v in metrics.items():
        is_scalar_namespace = (
            isinstance(v, Mapping)
            and k not in _KNOWN_ALIASES
            and not any(isinstance(x, (list, tuple, Mapping)) for x in v.values())
        )
        if is_scalar_namespace:
            for kk, vv in v.items():
                flat[kk] = vv
        else:
            flat[k] = v
    return flat


@dataclass
class FeatureSchema:
    """
    Esquema reproducible: define EXACTAMENTE qué columnas componen el vector y
    en qué orden, más las medianas para imputar valores faltantes.
    """

    scalar_names: list[str] = field(default_factory=lambda: list(SCALAR_ALIASES.keys()))
    vector_names: list[str] = field(default_factory=lambda: list(VECTOR_ALIASES.keys()))
    medians: dict[str, float] = field(default_factory=dict)
    # Opcionales para escalado consistente:
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    # ---- propiedades ----------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Nombre legible por columna del vector resultante."""
        out: list[str] = []
        for name in self.scalar_names:
            out.append(name)
            out.append(f"{name}__missing")
        for name in self.vector_names:
            for stat in VECTOR_STATS:
                out.append(f"{name}__{stat}")
            out.append(f"{name}__missing")
        return out

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    # ---- transformación principal --------------------------------------------

    def vectorize(self, metrics: Mapping[str, Any]) -> np.ndarray:
        flat = flatten_metrics(metrics)
        values: list[float] = []
        for name in self.scalar_names:
            raw = _first_match(flat, SCALAR_ALIASES[name])
            v = _as_float(raw)
            if v is None:
                values.append(self.medians.get(name, 0.0))
                values.append(1.0)            # missing = 1
            else:
                values.append(v)
                values.append(0.0)
        for name in self.vector_names:
            raw = _first_match(flat, VECTOR_ALIASES[name])
            arr = _as_array(raw)
            if arr is None:
                for stat in VECTOR_STATS:
                    values.append(self.medians.get(f"{name}__{stat}", 0.0))
                values.append(1.0)
            else:
                for v in _vector_stats(arr):
                    values.append(v)
                values.append(0.0)
        return np.asarray(values, dtype=np.float32)

    def vectorize_batch(self, batch: Iterable[Mapping[str, Any]]) -> np.ndarray:
        rows = [self.vectorize(m) for m in batch]
        return np.stack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        """Aplica z-score si hay means/stds. Si no, devuelve x sin cambios."""
        if not self.means or not self.stds:
            return x
        mean = np.asarray([self.means.get(n, 0.0) for n in self.feature_names], dtype=np.float32)
        std = np.asarray([self.stds.get(n, 1.0) for n in self.feature_names], dtype=np.float32)
        std = np.where(std > 1e-6, std, 1.0)
        return (x - mean) / std

    # ---- ajuste sobre un conjunto --------------------------------------------

    @classmethod
    def fit(cls, metrics_list: Iterable[Mapping[str, Any]]) -> "FeatureSchema":
        """
        Calcula medianas (para imputación) y mean/std (para z-score) ignorando
        los valores faltantes. Usa un pase doble: primero detecta presencia,
        luego computa estadísticos sobre los valores reales.
        """
        schema = cls()
        flats = [flatten_metrics(m) for m in metrics_list]

        # Recolecta valores reales por nombre.
        scalar_buf: dict[str, list[float]] = {n: [] for n in schema.scalar_names}
        vector_buf: dict[str, dict[str, list[float]]] = {
            n: {s: [] for s in VECTOR_STATS} for n in schema.vector_names
        }
        for flat in flats:
            for name in schema.scalar_names:
                v = _as_float(_first_match(flat, SCALAR_ALIASES[name]))
                if v is not None:
                    scalar_buf[name].append(v)
            for name in schema.vector_names:
                arr = _as_array(_first_match(flat, VECTOR_ALIASES[name]))
                if arr is not None:
                    for stat, val in zip(VECTOR_STATS, _vector_stats(arr)):
                        vector_buf[name][stat].append(val)

        # Medianas para imputación.
        for name, vals in scalar_buf.items():
            if vals:
                schema.medians[name] = float(np.median(vals))
        for name, stats in vector_buf.items():
            for stat, vals in stats.items():
                if vals:
                    schema.medians[f"{name}__{stat}"] = float(np.median(vals))

        # Mean/std sobre el conjunto vectorizado completo (incluye imputados y
        # banderas missing, que también participan en el z-score).
        if flats:
            X = schema.vectorize_batch(flats)
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            for n, m, s in zip(schema.feature_names, mean, std):
                schema.means[n] = float(m)
                schema.stds[n] = float(s)
        return schema

    # ---- persistencia --------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "scalar_names": self.scalar_names,
                    "vector_names": self.vector_names,
                    "medians": self.medians,
                    "means": self.means,
                    "stds": self.stds,
                    "feature_names": self.feature_names,
                    "dim": self.dim,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            scalar_names=list(data["scalar_names"]),
            vector_names=list(data["vector_names"]),
            medians=dict(data.get("medians", {})),
            means=dict(data.get("means", {})),
            stds=dict(data.get("stds", {})),
        )
