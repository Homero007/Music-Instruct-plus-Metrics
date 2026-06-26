"""
Metrica Kullback-Leibler Divergence (KLD) para comparar audio real vs generado.

La idea principal es:
1. Clasificar cada audio con un clasificador preentrenado.
2. Promediar las probabilidades por clase para obtener P y Q.
3. Calcular D_KL(P || Q).

Tambien puede ejecutarse desde consola usando archivos con probabilidades ya
calculadas (.npy, .csv o .json), por ejemplo:

    python kld_metric.py --real real_probs.npy --generated gen_probs.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


ProbVector = np.ndarray
Classifier = Callable[[str], Iterable[float]]


def _normalizar_distribucion(distribucion: Iterable[float], epsilon: float) -> np.ndarray:
    """Convierte una distribucion a probabilidades validas y evita ceros."""
    vector = np.asarray(distribucion, dtype=np.float64)

    if vector.ndim != 1:
        raise ValueError("La distribucion debe ser un vector de una dimension.")

    if np.any(vector < 0):
        raise ValueError("La distribucion no puede contener probabilidades negativas.")

    suma = vector.sum()
    if suma <= 0:
        raise ValueError("La distribucion debe tener suma mayor que 0.")

    vector = vector / suma
    vector = np.clip(vector, epsilon, None)
    return vector / vector.sum()


def promediar_predicciones(predicciones: Iterable[Iterable[float]], epsilon: float = 1e-12) -> np.ndarray:
    """
    Promedia una lista de vectores de probabilidad para obtener una distribucion global.

    Args:
        predicciones: matriz con forma (n_audios, n_clases).
        epsilon: valor minimo para evitar divisiones entre cero y log(0).

    Returns:
        Vector de probabilidad promedio con forma (n_clases,).
    """
    matriz = np.asarray(list(predicciones), dtype=np.float64)

    if matriz.ndim != 2:
        raise ValueError("Las predicciones deben tener forma (n_audios, n_clases).")

    if matriz.shape[0] == 0:
        raise ValueError("Se necesita al menos una prediccion.")

    if np.any(matriz < 0):
        raise ValueError("Las predicciones no pueden contener valores negativos.")

    filas_suma = matriz.sum(axis=1, keepdims=True)
    if np.any(filas_suma <= 0):
        raise ValueError("Cada vector de prediccion debe tener suma mayor que 0.")

    matriz = matriz / filas_suma
    distribucion = matriz.mean(axis=0)
    return _normalizar_distribucion(distribucion, epsilon)


def calcular_kld_desde_distribuciones(
    p_real: Iterable[float],
    q_generado: Iterable[float],
    epsilon: float = 1e-12,
) -> float:
    """
    Calcula D_KL(P || Q) entre dos distribuciones globales.

    KLD = 0 indica distribuciones identicas. Mientras mayor sea el valor,
    mayor es la diferencia estadistica entre audio real y generado.
    """
    p = _normalizar_distribucion(p_real, epsilon)
    q = _normalizar_distribucion(q_generado, epsilon)

    if p.shape != q.shape:
        raise ValueError(f"P y Q deben tener la misma forma. Recibido: {p.shape} y {q.shape}.")

    return float(np.sum(p * np.log(p / q)))


def calcular_kld(
    conjunto_real: Iterable[str],
    conjunto_generado: Iterable[str],
    clasificador: Classifier,
    epsilon: float = 1e-12,
) -> float:
    """
    Clasifica audios reales y generados, promedia sus probabilidades y calcula KLD.

    Args:
        conjunto_real: rutas de audios reales.
        conjunto_generado: rutas de audios generados.
        clasificador: funcion que recibe una ruta de audio y devuelve probabilidades por clase.
        epsilon: valor minimo para evitar log(0).

    Returns:
        D_KL(P_real || Q_generado).
    """
    p_lista = [clasificador(audio_r) for audio_r in conjunto_real]
    q_lista = [clasificador(audio_g) for audio_g in conjunto_generado]

    p = promediar_predicciones(p_lista, epsilon=epsilon)
    q = promediar_predicciones(q_lista, epsilon=epsilon)

    return calcular_kld_desde_distribuciones(p, q, epsilon=epsilon)


def cargar_predicciones(ruta: str | Path) -> np.ndarray:
    """
    Carga predicciones desde .npy, .csv o .json.

    Formato esperado:
        - Filas: audios.
        - Columnas: probabilidades por clase.
    """
    path = Path(ruta)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if path.suffix.lower() == ".npy":
        return np.load(path)

    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as archivo:
            return np.asarray(json.load(archivo), dtype=np.float64)

    raise ValueError("Formato no soportado. Usa .npy, .csv o .json.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula KLD entre probabilidades de audio real y generado."
    )
    parser.add_argument("--real", required=True, help="Archivo .npy, .csv o .json con predicciones reales.")
    parser.add_argument(
        "--generated",
        required=True,
        help="Archivo .npy, .csv o .json con predicciones generadas.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-12, help="Suavizado para evitar log(0).")
    args = parser.parse_args()

    pred_real = cargar_predicciones(args.real)
    pred_generado = cargar_predicciones(args.generated)

    p = promediar_predicciones(pred_real, epsilon=args.epsilon)
    q = promediar_predicciones(pred_generado, epsilon=args.epsilon)
    kld = calcular_kld_desde_distribuciones(p, q, epsilon=args.epsilon)

    print(f"KLD = {kld:.10f}")


if __name__ == "__main__":
    main()
