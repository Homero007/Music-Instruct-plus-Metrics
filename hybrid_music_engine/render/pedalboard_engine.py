from __future__ import annotations

import shutil
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.render.pedalboard_vst import process_audio_with_pedalboard


def render_midi_audio(
    midi_path: Path,
    output_dir: Path,
    *,
    config: EngineConfig,
    output_name: str = "render",
    engine: str = "auto",
    soundfont_path: Path | None = None,
    sample_rate: int = 44100,
    export_mp3: bool = False,
    gain: float = 0.18,
    pedalboard_preset: str = "master",
    plugin_paths: list[Path] | None = None,
) -> dict:
    source = Path(midi_path).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"MIDI no encontrado: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{output_name}.wav"
    requested_engine = engine

    if engine not in {"auto", "preview", "fluidsynth", "pedalboard"}:
        raise RuntimeError("engine debe ser auto, preview, fluidsynth o pedalboard.")

    if engine == "auto":
        candidate_soundfont = (soundfont_path or config.default_soundfont_path).expanduser()
        has_fluidsynth = shutil.which(config.fluidsynth_binary) is not None
        engine = "fluidsynth" if has_fluidsynth and candidate_soundfont.exists() else "preview"

    if engine == "pedalboard":
        base_payload = render_midi_fluidsynth_wav(
            source,
            output_dir / f"{output_name}.raw.wav",
            config=config,
            soundfont_path=soundfont_path,
            sample_rate=sample_rate,
        )
        render_payload = process_audio_with_pedalboard(
            Path(str(base_payload["wav_path"])),
            wav_path,
            plugin_paths=plugin_paths,
            preset=pedalboard_preset,
        )
        render_payload["source_midi"] = str(source)
        render_payload["instrument_render"] = base_payload
    elif engine == "fluidsynth":
        render_payload = render_midi_fluidsynth_wav(
            source,
            wav_path,
            config=config,
            soundfont_path=soundfont_path,
            sample_rate=sample_rate,
        )
    else:
        render_payload = render_midi_preview_wav(
            source,
            wav_path,
            sample_rate=sample_rate,
            gain=gain,
        )

    render_payload["requested_engine"] = requested_engine
    render_payload["engine"] = render_payload.get("engine", engine)
    if export_mp3:
        mp3_path = output_dir / f"{output_name}.mp3"
        render_payload["mp3_path"] = str(convert_wav_to_mp3(wav_path, mp3_path, config=config))
    metadata_path = output_dir / "render.json"
    render_payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    render_payload["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(render_payload, indent=2), encoding="utf-8")
    return render_payload


def render_midi_fluidsynth_wav(
    midi_path: Path,
    output_wav: Path,
    *,
    config: EngineConfig,
    soundfont_path: Path | None = None,
    sample_rate: int = 44100,
) -> dict:
    fluidsynth = shutil.which(config.fluidsynth_binary)
    if not fluidsynth:
        raise RuntimeError(
            "FluidSynth no está disponible. Instálalo o usa engine='preview'."
        )
    soundfont = (soundfont_path or config.default_soundfont_path).expanduser().resolve()
    if not soundfont.exists():
        raise RuntimeError(
            f"SoundFont no encontrado: {soundfont}. Coloca uno en assets/soundfonts/default.sf2 "
            "o define HYBRID_SOUNDFONT_PATH."
        )
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        fluidsynth,
        "-ni",
        "-F",
        str(output_wav),
        "-r",
        str(sample_rate),
        str(soundfont),
        str(midi_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "FluidSynth falló."
        raise RuntimeError(message)
    return {
        "source_midi": str(Path(midi_path).expanduser().resolve()),
        "wav_path": str(output_wav),
        "sample_rate": sample_rate,
        "engine": "fluidsynth",
        "soundfont_path": str(soundfont),
    }


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, *, config: EngineConfig) -> Path:
    ffmpeg = shutil.which(config.ffmpeg_binary)
    if not ffmpeg:
        raise RuntimeError("FFmpeg no está disponible. Instálalo para exportar MP3.")
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(mp3_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "FFmpeg falló."
        raise RuntimeError(message)
    return mp3_path


def mix_layer_renders(
    renders: dict[str, dict],
    output_dir: Path,
    *,
    config: EngineConfig,
    output_name: str = "mix",
    export_mp3: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("La mezcla requiere numpy y soundfile.") from exc

    layer_settings = {
        "drums": {"gain": 0.92, "pan": 0.0},
        "bass": {"gain": 0.82, "pan": 0.0},
        "harmony": {"gain": 0.72, "pan": -0.25},
        "melody": {"gain": 0.88, "pan": 0.2},
    }
    loaded: list[tuple[str, Any, int]] = []
    max_samples = 0
    sample_rate = 44100
    for layer_name, payload in renders.items():
        wav_path = Path(str(payload.get("wav_path", ""))).expanduser()
        if not wav_path.exists():
            continue
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        sample_rate = int(sr)
        loaded.append((layer_name, audio, sample_rate))
        max_samples = max(max_samples, audio.shape[0])
    if not loaded:
        raise RuntimeError("No hay WAVs de capas para mezclar.")

    mix = np.zeros((max_samples, 2), dtype=np.float32)
    for layer_name, audio, _sr in loaded:
        settings = layer_settings.get(layer_name, {"gain": 0.75, "pan": 0.0})
        stereo = audio[:, :2] if audio.shape[1] >= 2 else np.repeat(audio[:, :1], 2, axis=1)
        padded = np.zeros((max_samples, 2), dtype=np.float32)
        padded[: stereo.shape[0], :] = stereo
        gain = float(settings["gain"])
        pan = max(min(float(settings["pan"]), 1.0), -1.0)
        left_gain = gain * (1.0 - max(pan, 0.0) * 0.45)
        right_gain = gain * (1.0 + min(pan, 0.0) * 0.45)
        padded[:, 0] *= left_gain
        padded[:, 1] *= right_gain
        mix += padded

    mix = _master_limiter(mix)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{output_name}.wav"
    sf.write(wav_path, mix, sample_rate, subtype="PCM_24")
    payload = {
        "engine": "layer-mix-master",
        "wav_path": str(wav_path),
        "sample_rate": sample_rate,
        "layers": sorted(renders.keys()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if export_mp3:
        mp3_path = output_dir / f"{output_name}.mp3"
        payload["mp3_path"] = str(convert_wav_to_mp3(wav_path, mp3_path, config=config))
    metadata_path = output_dir / "render.json"
    payload["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _master_limiter(audio):
    import numpy as np

    if audio.size == 0:
        return audio
    audio = np.tanh(audio * 0.9)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / max(peak / 0.98, 1.0)
    return audio.astype(np.float32)


def render_midi_preview_wav(
    midi_path: Path,
    output_wav: Path,
    *,
    sample_rate: int = 44100,
    gain: float = 0.18,
) -> dict:
    try:
        import numpy as np
        import soundfile as sf
        from mido import MidiFile, tick2second
    except ImportError as exc:
        raise RuntimeError("Render preview requiere numpy, soundfile y mido.") from exc

    source = Path(midi_path).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"MIDI no encontrado: {source}")

    midi = MidiFile(source)
    tempo = 500000
    notes: list[tuple[int, float, float, int]] = []
    max_time = 0.0
    for track in midi.tracks:
        absolute_ticks = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for message in track:
            absolute_ticks += int(message.time)
            if message.type == "set_tempo":
                tempo = int(message.tempo)
            seconds = float(tick2second(absolute_ticks, midi.ticks_per_beat, tempo))
            max_time = max(max_time, seconds)
            if message.type == "note_on" and message.velocity > 0:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                active.setdefault(key, []).append((absolute_ticks, int(message.velocity)))
            elif message.type in {"note_off", "note_on"}:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                starts = active.get(key)
                if not starts:
                    continue
                start_tick, velocity = starts.pop(0)
                start = float(tick2second(start_tick, midi.ticks_per_beat, tempo))
                end = max(seconds, start + 0.05)
                notes.append((key[1], start, end, velocity))
                max_time = max(max_time, end)

    total_samples = max(int((max_time + 0.5) * sample_rate), sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    for pitch, start, end, velocity in notes:
        start_idx = max(int(start * sample_rate), 0)
        end_idx = min(max(int(end * sample_rate), start_idx + 1), total_samples)
        length = end_idx - start_idx
        t = np.arange(length, dtype=np.float32) / sample_rate
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        envelope = np.linspace(1.0, 0.15, length, dtype=np.float32)
        audio[start_idx:end_idx] += (
            np.sin(2 * np.pi * freq * t).astype(np.float32)
            * envelope
            * gain
            * max(min(velocity / 127.0, 1.0), 0.05)
        )

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, audio, sample_rate, subtype="PCM_24")
    return {
        "source_midi": str(source),
        "wav_path": str(output_wav),
        "sample_rate": sample_rate,
        "duration_seconds": round(total_samples / sample_rate, 4),
        "note_count": len(notes),
        "engine": "internal-sine-preview",
    }


def render_with_pedalboard_fallback(midi_path: Path, plugin_path: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    render_midi_preview_wav(midi_path, output_wav)
    return output_wav
