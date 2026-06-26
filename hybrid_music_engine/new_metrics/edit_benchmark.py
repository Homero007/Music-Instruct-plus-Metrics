#!/usr/bin/env python3
"""edit_benchmark.py — Benchmark de EDICIÓN por instrucciones.

Motivación
----------
El benchmark de generación (texto -> audio sobre MusicCaps; ver
``benchmark_analysis.py``) compara modelos en generación libre y justifica la
elección del *modelo base*. **No** evidencia capacidad de edición: un buen
CLAP en generación libre no implica preservar identidad, instrumentación,
estructura o mezcla de una pista FUENTE al aplicar una operación pedida.

Este módulo evalúa la tarea real con ternas
``{fuente, instrucción, objetivo}`` y la SALIDA del editor, con 5 métricas:

  - **FAD-to-target**      distancia de Fréchet (salidas vs objetivos). Menor mejor.
  - **CLAP-instruction**   coseno(salida, instrucción) por clip. Mayor mejor.
  - **Content-preservation** similitud(fuente, salida) en lo NO editado. Mayor mejor.
  - **Operation-success**  fracción del cambio fuente->objetivo lograda. Mayor mejor.
  - **Human-preference**   placeholder para escucha controlada (MUSHRA).

Líneas base recomendadas (para no confundir generación con edición):
  1. Reconstrucción sin edición (salida = fuente): preserva todo, opera 0.
  2. MusicGen condicionado solo por texto (ignora la fuente).
  3. MusicGen con audio prompt SIN módulo de fusión.
  4. Referencia más cercana: Instruct-MusicGen (o el adapter con AudioFusion).

Las funciones operan sobre embeddings/escalares (NumPy) para ser deterministas
y testeables; los extractores de audio (VGGish/CLAP) se inyectan aparte y
reutilizan los de ``fad_score``/``clap_score`` (mismo espacio, sin drift).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Reutiliza la matemática ya validada de los otros módulos (sin reimplementar).
try:  # como paquete
    from .fad_score import compute_statistics, frechet_distance
    from .clap_score import cosine_similarity
except ImportError:  # ejecución suelta (sys.path incluye este dir)
    from fad_score import compute_statistics, frechet_distance
    from clap_score import cosine_similarity


# 4 métricas automáticas + dirección (la 5ª, preferencia humana, es externa).
EDIT_METRICS: dict[str, str] = {
    "fad_to_target": "min",
    "clap_instruction": "max",
    "content_preservation": "max",
    "operation_success": "max",
}
EDIT_LABELS = {
    "fad_to_target": "FAD→objetivo",
    "clap_instruction": "CLAP-instrucción",
    "content_preservation": "Preservación",
    "operation_success": "Éxito de operación",
}


# ── Métricas individuales ────────────────────────────────────────────────────

def fad_to_target(output_embs: np.ndarray, target_embs: np.ndarray) -> float:
    """FAD entre el conjunto de SALIDAS y el de OBJETIVOS (menor = más cercano)."""
    mu_o, sig_o = compute_statistics(np.asarray(output_embs, dtype=np.float64))
    mu_t, sig_t = compute_statistics(np.asarray(target_embs, dtype=np.float64))
    return frechet_distance(mu_t, sig_t, mu_o, sig_o)


def clap_instruction_score(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Media de coseno(embedding_audio_salida, embedding_texto_instrucción)."""
    sims = [cosine_similarity(np.asarray(a), np.asarray(t)) for a, t in pairs]
    return float(np.mean(sims)) if sims else float("nan")


def content_preservation(source_emb: np.ndarray, output_emb: np.ndarray) -> float:
    """Similitud coseno fuente↔salida (1 = preserva el contenido de la fuente).

    Proxy global. Para edición fina conviene la variante por stems, que mide la
    preservación SOLO en las capas que no debían cambiar (ver abajo).
    """
    return float(cosine_similarity(np.asarray(source_emb), np.asarray(output_emb)))


def content_preservation_stems(
    source_stems: dict[str, np.ndarray],
    output_stems: dict[str, np.ndarray],
    edited_keys: set[str],
) -> float:
    """Preservación medida solo en los stems NO editados (media de coseno).

    `edited_keys` son las capas que la operación SÍ debe cambiar (p. ej. añadir
    batería -> {"drums"}); el resto debe conservarse. Devuelve NaN si todo se
    edita (no hay nada que preservar).
    """
    preserved = [k for k in source_stems if k not in edited_keys and k in output_stems]
    if not preserved:
        return float("nan")
    sims = [cosine_similarity(source_stems[k], output_stems[k]) for k in preserved]
    return float(np.mean(sims))


def operation_success(
    source_score: float,
    output_score: float,
    target_score: float,
    eps: float = 1e-6,
) -> float:
    """Fracción del cambio fuente->objetivo lograda por la salida, recortada a [0,1].

    `*_score` es la medida del atributo objetivo de la operación (p. ej. energía
    de batería para "añade batería", o probabilidad de voz para "quita la voz").

    - Reconstrucción (salida = fuente) -> 0.0 (no aplicó la operación).
    - Salida = objetivo                -> 1.0.
    - Sobrepaso                         -> recortado a 1.0.
    - Si no se requería cambio (objetivo ≈ fuente) -> qué tan cerca quedó de la fuente.
    """
    denom = float(target_score) - float(source_score)
    if abs(denom) < eps:
        return float(max(0.0, 1.0 - abs(float(output_score) - float(source_score))))
    frac = (float(output_score) - float(source_score)) / denom
    return float(min(1.0, max(0.0, frac)))


# ── Agregación por modelo / línea base ───────────────────────────────────────

def compute_edit_table(per_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Tabla comparativa de edición + ranking global.

    `per_model[name]` debe contener:
        output_embs, target_embs          : (N, D) para FAD-to-target
        clap_scores                        : lista de coseno(salida, instrucción)
        content_preservation               : lista de preservación por terna
        operation_success                  : lista de éxito por terna
    """
    rows: list[dict[str, Any]] = []
    for name, d in per_model.items():
        rows.append({
            "model": name,
            "fad_to_target": round(fad_to_target(d["output_embs"], d["target_embs"]), 4),
            "clap_instruction": round(float(np.mean(d["clap_scores"])), 4),
            "content_preservation": round(float(np.mean(d["content_preservation"])), 4),
            "operation_success": round(float(np.mean(d["operation_success"])), 4),
        })

    def _key(value: float, direction: str) -> float:
        # NaN (p. ej. CLAP no calculado) se ordena al final (peor).
        if value != value:
            return float("-inf") if direction == "max" else float("inf")
        return value

    models = [r["model"] for r in rows]
    ranks: dict[str, dict[str, float]] = {}
    for metric, direction in EDIT_METRICS.items():
        keyed = {m: _key(_row_for(rows, m)[metric], direction) for m in models}
        order = sorted(models, key=lambda m: keyed[m], reverse=(direction == "max"))
        # Rangos PROMEDIO para empates: los modelos con el mismo valor (p. ej.
        # CLAP=NaN para todos) comparten rango y no se desempatan arbitrariamente.
        rank_map: dict[str, float] = {}
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and keyed[order[j + 1]] == keyed[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rank_map[order[k]] = avg_rank
            i = j + 1
        ranks[metric] = rank_map
    for r in rows:
        r["overall_rank"] = float(np.mean([ranks[m][r["model"]] for m in EDIT_METRICS]))
    rows.sort(key=lambda r: r["overall_rank"])
    return {"rows": rows, "metrics": dict(EDIT_METRICS), "labels": dict(EDIT_LABELS)}


def _row_for(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return next(r for r in rows if r["model"] == model)
