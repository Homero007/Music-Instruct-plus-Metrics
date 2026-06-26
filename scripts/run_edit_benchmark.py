#!/usr/bin/env python3
"""run_edit_benchmark.py — CLI end-to-end del benchmark de EDICIÓN.

Toma el manifest de ternas (``build_edit_triplets.py``) y las SALIDAS de uno o
más modelos / líneas base, extrae embeddings y calcula las 5 métricas de
``edit_benchmark.py`` en una tabla comparativa.

Embeddings: por defecto log-mel (librosa, determinista, sin descargas) para
FAD-to-target y preservación. Con ``--clap`` usa laion-clap para la métrica
CLAP-instrucción (requiere el paquete). El éxito de operación se mide con el
atributo del manifest (BPM/RMS/energía de cola) sobre la salida.

Líneas base:
  --reconstruction        añade la baseline reconstrucción (salida = fuente).
  --models name=dir ...   salidas de cada modelo en dir/<id>.wav.

Uso:
  python run_edit_benchmark.py --manifest edit_triplets/edit_triplets.csv \
      --reconstruction --models instruct=outputs/instruct \
      --out results/edit_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_music_engine.new_metrics import edit_benchmark as eb
from hybrid_music_engine.new_metrics.clap_score import cosine_similarity
import build_edit_triplets as bt

SR = 32000
N_MELS = 64


# ── Carga y embeddings ───────────────────────────────────────────────────────

def load_audio(path: Path, sr_target: int = SR) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    if sr != sr_target:
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr_target).astype(np.float64)
    return bt.fit_length(y)


def mel_embedding(y: np.ndarray, sr: int = SR, n_mels: int = N_MELS) -> np.ndarray:
    """Vector log-mel promediado en el tiempo (determinista, sin modelo)."""
    import librosa
    S = librosa.feature.melspectrogram(y=np.asarray(y, dtype=np.float32), sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S + 1e-10).mean(axis=1)


class ClapBackend:
    def __init__(self) -> None:
        import laion_clap
        self.m = laion_clap.CLAP_Module(enable_fusion=False)
        self.m.load_ckpt()

    def audio(self, y: np.ndarray) -> np.ndarray:
        return self.m.get_audio_embedding_from_data(x=np.asarray(y)[None, :], use_tensor=False)[0]

    def text(self, t: str) -> np.ndarray:
        return self.m.get_text_embedding([t], use_tensor=False)[0]


# ── Evaluación por modelo ────────────────────────────────────────────────────

def reconstruction_output(row: dict) -> np.ndarray:
    """Ancla inferior: no edita (salida = fuente)."""
    return load_audio(Path(row["source_path"]))


def oracle_output(row: dict) -> np.ndarray:
    """Ancla superior: edición perfecta (salida = objetivo)."""
    return load_audio(Path(row["target_path"]))


def dir_output(out_dir: Path):
    def get(row: dict) -> np.ndarray:
        return load_audio(Path(out_dir) / f"{row['id']}.wav")
    return get


def evaluate_model(name: str, rows: list[dict], get_output, clap: ClapBackend | None = None) -> dict:
    out_embs, tgt_embs = [], []
    clap_scores, preservation, op_success = [], [], []
    for row in rows:
        out_y = get_output(row)
        tgt_y = load_audio(Path(row["target_path"]))
        src_y = load_audio(Path(row["source_path"]))
        oe, te, se = mel_embedding(out_y), mel_embedding(tgt_y), mel_embedding(src_y)
        out_embs.append(oe)
        tgt_embs.append(te)
        preservation.append(cosine_similarity(se, oe))

        sa, ta = row.get("source_attr", ""), row.get("target_attr", "")
        if str(sa) not in ("", "None") and str(ta) not in ("", "None"):
            out_attr = bt.ATTR_FNS[row["attr_name"]](out_y, SR)
            op_success.append(eb.operation_success(float(sa), out_attr, float(ta)))

        if clap is not None:
            clap_scores.append(cosine_similarity(clap.audio(out_y), clap.text(row["instruction"])))

    return {
        "output_embs": np.array(out_embs),
        "target_embs": np.array(tgt_embs),
        "clap_scores": clap_scores or [float("nan")],
        "content_preservation": preservation,
        "operation_success": op_success or [float("nan")],
        "n_triplets": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark de edición end-to-end.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV de ternas (build_edit_triplets).")
    parser.add_argument("--models", nargs="*", default=[], help="name=dir con salidas dir/<id>.wav.")
    parser.add_argument("--reconstruction", action="store_true", help="Ancla inferior: reconstrucción (salida=fuente, no edita).")
    parser.add_argument("--oracle", action="store_true", help="Ancla superior: oráculo (salida=objetivo, edición perfecta).")
    parser.add_argument("--clap", action="store_true", help="Calcula CLAP-instrucción con laion-clap.")
    parser.add_argument("--out", type=Path, default=Path("results/edit_eval"), help="Carpeta de salida.")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if not rows:
        print("Manifest vacío.")
        return 1

    clap = None
    if args.clap:
        try:
            clap = ClapBackend()
        except Exception as exc:  # noqa: BLE001
            print(f"[aviso] CLAP no disponible ({exc}); se omite CLAP-instrucción.")

    per_model: dict[str, dict] = {}
    if args.oracle:
        print("Evaluando ancla superior: oracle (salida=objetivo)...")
        per_model["oracle"] = evaluate_model("oracle", rows, oracle_output, clap)
    if args.reconstruction:
        print("Evaluando ancla inferior: reconstruction (salida=fuente)...")
        per_model["reconstruction"] = evaluate_model("reconstruction", rows, reconstruction_output, clap)
    for spec in args.models:
        name, _, path = spec.partition("=")
        print(f"Evaluando modelo: {name} ({path})...")
        per_model[name] = evaluate_model(name, rows, dir_output(Path(path)), clap)

    if not per_model:
        print("Nada que evaluar: usa --oracle, --reconstruction o --models name=dir.")
        return 1

    table = eb.compute_edit_table(per_model)
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["model", "fad_to_target", "clap_instruction", "content_preservation", "operation_success", "overall_rank"]
    with open(args.out / "edit_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table["rows"]:
            w.writerow({c: r[c] for c in cols})
    (args.out / "edit_comparison.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Tabla de edición (ranking) ===")
    print(f"{'Modelo':22} {'FAD↓':>9} {'CLAP↑':>8} {'Preserv↑':>9} {'OpÉxito↑':>9} {'Rango':>6}")
    for r in table["rows"]:
        print(f"{r['model']:22} {r['fad_to_target']:>9.4f} {r['clap_instruction']:>8} "
              f"{r['content_preservation']:>9} {r['operation_success']:>9} {r['overall_rank']:>6.2f}")
    print(f"\nResultados en: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
