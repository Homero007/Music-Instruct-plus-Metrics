from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    *,
    model_name: str = "htdemucs",
    device: str = "auto",
) -> dict[str, Any]:
    if not audio_path.exists():
        raise RuntimeError(f"Audio para separación no encontrado: {audio_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from demucs.api import Separator
    except (ImportError, ModuleNotFoundError):
        return _separate_stems_with_cli(
            audio_path,
            output_dir,
            model_name=model_name,
            device=device,
        )

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile es necesario para escribir stems WAV.") from exc

    resolved_device = _resolve_device(device)
    separator = Separator(model=model_name, device=resolved_device)
    _origin, separated = separator.separate_audio_file(str(audio_path))

    stems: dict[str, str] = {}
    sample_rate = int(getattr(separator, "samplerate", 44100))
    for name, audio in separated.items():
        stem_path = output_dir / f"{name}.wav"
        array = _stem_to_numpy(audio)
        sf.write(stem_path, array, sample_rate, subtype="PCM_24")
        stems[name] = str(stem_path)

    expected = {"drums", "bass", "vocals", "other"}
    missing = sorted(expected - set(stems))
    return {
        "source": str(audio_path),
        "model": model_name,
        "device": resolved_device,
        "sample_rate": sample_rate,
        "files": stems,
        "missing": missing,
        "engine": "demucs-api",
    }


def _separate_stems_with_cli(
    audio_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    device: str,
) -> dict[str, Any]:
    try:
        import demucs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Demucs no está instalado. Instala las dependencias opcionales con "
            "`python -m pip install -e \".[audio]\"` dentro de hybrid_engine."
        ) from exc

    resolved_device = _resolve_device(device)
    demucs_output = output_dir / ".demucs"
    if demucs_output.exists():
        shutil.rmtree(demucs_output)
    command = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model_name,
        "-d",
        resolved_device,
        "-o",
        str(demucs_output),
        "--filename",
        "{stem}.{ext}",
        "--int24",
        str(audio_path),
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(_torch_cache_dir(output_dir)))
    Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Demucs falló al separar stems. {details}")

    stems: dict[str, str] = {}
    for generated in sorted((demucs_output / model_name).glob("*.wav")):
        stem_name = generated.stem
        stem_path = output_dir / f"{stem_name}.wav"
        shutil.move(str(generated), stem_path)
        stems[stem_name] = str(stem_path)
    shutil.rmtree(demucs_output, ignore_errors=True)

    expected = {"drums", "bass", "vocals", "other"}
    missing = sorted(expected - set(stems))
    if not stems:
        raise RuntimeError("Demucs terminó sin generar stems WAV.")
    return {
        "source": str(audio_path),
        "model": model_name,
        "device": resolved_device,
        "sample_rate": None,
        "files": stems,
        "missing": missing,
        "engine": "demucs-cli",
    }


def _torch_cache_dir(output_dir: Path) -> Path:
    for parent in [output_dir, *output_dir.parents]:
        if parent.name == "data":
            return parent / "model_cache" / "torch"
    return output_dir / ".torch_cache"


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stem_to_numpy(audio):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("numpy y torch son necesarios para convertir stems.") from exc

    if isinstance(audio, torch.Tensor):
        array = audio.detach().cpu().float().numpy()
    else:
        array = np.asarray(audio, dtype=np.float32)

    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.T if array.shape[0] <= array.shape[1] else array
    raise RuntimeError(f"Stem con forma no soportada: {array.shape}")
