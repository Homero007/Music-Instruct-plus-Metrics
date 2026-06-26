from __future__ import annotations

from pathlib import Path
from typing import Any


def process_audio_with_pedalboard(
    input_wav: Path,
    output_wav: Path,
    *,
    plugin_paths: list[Path] | None = None,
    preset: str = "master",
) -> dict[str, Any]:
    try:
        import soundfile as sf
        from pedalboard import (
            Compressor,
            Gain,
            HighpassFilter,
            Limiter,
            LowpassFilter,
            Pedalboard,
            load_plugin,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Pedalboard no está instalado. Instala dependencias opcionales con "
            ".venv/bin/python -m pip install -e '.[audio]'."
        ) from exc

    source = Path(input_wav).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"WAV no encontrado: {source}")
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    plugins = _preset_plugins(preset, Compressor, Gain, HighpassFilter, Limiter, LowpassFilter)
    loaded_paths: list[str] = []
    for plugin_path in plugin_paths or []:
        resolved = Path(plugin_path).expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"Plugin no encontrado: {resolved}")
        plugins.append(load_plugin(str(resolved)))
        loaded_paths.append(str(resolved))
    board = Pedalboard(plugins)
    processed = board(audio, sample_rate)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, processed, sample_rate, subtype="PCM_24")
    return {
        "engine": "pedalboard-effects",
        "input_wav": str(source),
        "wav_path": str(output_wav),
        "sample_rate": int(sample_rate),
        "preset": preset,
        "plugins": loaded_paths,
    }


def _preset_plugins(preset: str, compressor, gain, highpass, limiter, lowpass) -> list[Any]:
    if preset == "none":
        return []
    if preset == "clean":
        return [
            highpass(cutoff_frequency_hz=25.0),
            compressor(threshold_db=-16.0, ratio=2.0),
            gain(gain_db=0.8),
            limiter(threshold_db=-1.0),
        ]
    if preset == "warm":
        return [
            highpass(cutoff_frequency_hz=35.0),
            lowpass(cutoff_frequency_hz=15500.0),
            compressor(threshold_db=-20.0, ratio=2.4),
            gain(gain_db=1.2),
            limiter(threshold_db=-1.0),
        ]
    if preset == "punchy":
        return [
            highpass(cutoff_frequency_hz=32.0),
            compressor(threshold_db=-24.0, ratio=4.0),
            gain(gain_db=2.2),
            limiter(threshold_db=-0.9),
        ]
    if preset == "bright":
        return [
            highpass(cutoff_frequency_hz=40.0),
            lowpass(cutoff_frequency_hz=18500.0),
            compressor(threshold_db=-17.0, ratio=2.2),
            gain(gain_db=1.8),
            limiter(threshold_db=-0.8),
        ]
    if preset == "streaming":
        return [
            highpass(cutoff_frequency_hz=30.0),
            compressor(threshold_db=-18.0, ratio=2.8),
            gain(gain_db=1.0),
            limiter(threshold_db=-1.2),
        ]
    return [
        highpass(cutoff_frequency_hz=28.0),
        compressor(threshold_db=-18.0, ratio=3.0),
        gain(gain_db=1.5),
        limiter(threshold_db=-0.8),
    ]


def explain_vst_instrument_limit() -> str:
    return (
        "El render MIDI->instrumento VST3 requiere un plugin instrumento compatible y un host que "
        "acepte eventos MIDI. En esta fase se soporta Pedalboard para procesar audio renderizado "
        "con plugins de efecto; el instrumento principal sigue entrando por FluidSynth/preview."
    )
