import os
import gc
import csv
import torch
import torchaudio
import logging
from pathlib import Path

# Setup logging para incidencias
logging.basicConfig(
    filename='incidencias.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes de Generación
TARGET_DURATION = 10.0
TARGET_SR = 32000
TARGET_DBFS = -3.0

def peak_normalize(waveform: torch.Tensor, target_dbfs: float) -> torch.Tensor:
    """Normaliza los picos de la forma de onda al nivel especificado en dBFS."""
    peak = waveform.abs().max()
    if peak > 0:
        target_linear = 10 ** (target_dbfs / 20.0)
        return (waveform / peak) * target_linear
    return waveform

def clear_vram():
    """Limpia la VRAM y fuerza la recolección de basura."""
    gc.collect()
    torch.cuda.empty_cache()

def resample_if_needed(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr).to(waveform.device)
        return resampler(waveform)
    return waveform

def generate_musicgen(model_id, metadata, out_dir, device="cuda"):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    
    print(f"\nCargando {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device)
    
    # 10s target
    max_new_tokens = int(256 * (TARGET_DURATION / 5.0)) # MusicGen produces ~256 tokens per 5s
    
    for row in metadata:
        uid = row['id']
        caption = row['caption']
        out_path = out_dir / f"{uid}.wav"
        if out_path.exists():
            continue
            
        print(f"[{model_id}] Generando {uid}...")
        try:
            inputs = processor(text=[caption], padding=True, return_tensors="pt").to(device)
            # Semilla fija
            torch.manual_seed(42)
            
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=250,
                top_p=0.0
            )
            
            waveform = audio_values[0, 0].cpu()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Post-procesamiento
            waveform = peak_normalize(waveform, TARGET_DBFS)
            torchaudio.save(out_path, waveform, TARGET_SR)
            
        except Exception as e:
            logging.warning(f"Fallo en {model_id} - {uid}: {str(e)}. Re-intentando con semilla 43.")
            try:
                torch.manual_seed(43)
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_k=250,
                    top_p=0.0
                )
                waveform = audio_values[0, 0].cpu()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                waveform = peak_normalize(waveform, TARGET_DBFS)
                torchaudio.save(out_path, waveform, TARGET_SR)
            except Exception as e2:
                logging.error(f"Fallo definitivo en {model_id} - {uid}: {str(e2)}")

    del model
    del processor
    clear_vram()

def generate_audioldm2(model_id, metadata, out_dir, device="cuda"):
    from diffusers import AudioLDM2Pipeline
    
    print(f"\nCargando {model_id}...")
    try:
        pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    except Exception as e:
        print(f"No se pudo cargar AudioLDM2: {e}")
        return
        
    for row in metadata:
        uid = row['id']
        caption = row['caption']
        out_path = out_dir / f"{uid}.wav"
        if out_path.exists():
            continue
            
        print(f"[{model_id}] Generando {uid}...")
        try:
            generator = torch.Generator(device).manual_seed(42)
            audio = pipe(
                caption,
                num_inference_steps=200,
                audio_length_in_s=TARGET_DURATION,
                guidance_scale=3.5,
                generator=generator
            ).audios[0]
            
            waveform = torch.tensor(audio).unsqueeze(0) # [1, T]
            
            # Resample 16k -> 32k
            waveform = resample_if_needed(waveform, 16000, TARGET_SR)
            waveform = peak_normalize(waveform, TARGET_DBFS)
            torchaudio.save(out_path, waveform, TARGET_SR)
            
        except Exception as e:
            logging.warning(f"Fallo en {model_id} - {uid}: {str(e)}. Re-intentando con semilla 43.")
            try:
                generator = torch.Generator(device).manual_seed(43)
                audio = pipe(
                    caption,
                    num_inference_steps=200,
                    audio_length_in_s=TARGET_DURATION,
                    guidance_scale=3.5,
                    generator=generator
                ).audios[0]
                
                waveform = torch.tensor(audio).unsqueeze(0)
                waveform = resample_if_needed(waveform, 16000, TARGET_SR)
                waveform = peak_normalize(waveform, TARGET_DBFS)
                torchaudio.save(out_path, waveform, TARGET_SR)
            except Exception as e2:
                logging.error(f"Fallo definitivo en {model_id} - {uid}: {str(e2)}")

    del pipe
    clear_vram()

def generate_stable_audio(model_id, metadata, out_dir, device="cuda"):
    """Stable Audio Open (stabilityai/stable-audio-open-1.0).

    Genera audio estéreo a 44 100 Hz y lo convierte a mono 32 kHz.
    En MPS se fuerza float32 porque MPS no soporta todas las ops en float16.
    """
    from diffusers import StableAudioPipeline

    print(f"\nCargando {model_id}...")
    try:
        dtype = torch.float32 if device == "mps" else torch.float16
        pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"No se pudo cargar Stable Audio Open: {e}")
        logging.warning(f"Saltando stable-audio-open: {e}")
        return

    SAO_SR = 44100  # tasa nativa del modelo

    for row in metadata:
        uid = row['id']
        caption = row['caption']
        out_path = out_dir / f"{uid}.wav"
        if out_path.exists():
            continue

        print(f"[{model_id}] Generando {uid}...")
        try:
            generator = torch.Generator(device).manual_seed(42)
            result = pipe(
                caption,
                negative_prompt="Low quality, noise, distortion.",
                num_inference_steps=200,
                audio_end_in_s=TARGET_DURATION,
                num_waveforms_per_prompt=1,
                generator=generator,
            )
            audio = result.audios[0]  # [channels, samples] numpy float32

            waveform = torch.from_numpy(audio)           # [C, T]
            waveform = waveform.mean(dim=0, keepdim=True)  # mono [1, T]
            waveform = resample_if_needed(waveform.cpu(), SAO_SR, TARGET_SR)
            waveform = peak_normalize(waveform, TARGET_DBFS)
            torchaudio.save(out_path, waveform, TARGET_SR)

        except Exception as e:
            logging.warning(f"Fallo en {model_id} - {uid}: {str(e)}. Re-intentando con semilla 43.")
            try:
                generator = torch.Generator(device).manual_seed(43)
                result = pipe(
                    caption,
                    negative_prompt="Low quality, noise, distortion.",
                    num_inference_steps=200,
                    audio_end_in_s=TARGET_DURATION,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                )
                audio = result.audios[0]
                waveform = torch.from_numpy(audio)
                waveform = waveform.mean(dim=0, keepdim=True)
                waveform = resample_if_needed(waveform.cpu(), SAO_SR, TARGET_SR)
                waveform = peak_normalize(waveform, TARGET_DBFS)
                torchaudio.save(out_path, waveform, TARGET_SR)
            except Exception as e2:
                logging.error(f"Fallo definitivo en {model_id} - {uid}: {str(e2)}")

    del pipe
    clear_vram()


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Genera audio con los 4 modelos sobre testset_metadata.csv")
    parser.add_argument("--models", nargs="+",
                        choices=["musicgen-small", "musicgen-medium", "audioldm2", "stable-audio-open"],
                        default=["musicgen-small", "musicgen-medium"],
                        help="Modelos a ejecutar")
    parser.add_argument("--device", default=None,
                        help="Dispositivo: mps | cuda | cpu. Auto-detectado si no se indica.")
    parser.add_argument("--wavs-dir", type=Path, default=None,
                        help="Carpeta de salida (por defecto: <proyecto>/wavs/)")
    args = parser.parse_args()

    device = args.device or detect_device()
    root_dir = Path(__file__).resolve().parent.parent
    csv_path = root_dir / "testset_metadata.csv"
    wavs_dir = args.wavs_dir or root_dir / "wavs"

    if not csv_path.exists():
        print(f"Error: No se encontró {csv_path}")
        return

    metadata = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    print(f"Cargados {len(metadata)} registros de prueba.")
    print(f"Dispositivo: {device}")

    if "musicgen-small" in args.models:
        out = wavs_dir / "musicgen-small"
        out.mkdir(parents=True, exist_ok=True)
        generate_musicgen("facebook/musicgen-small", metadata, out, device=device)

    if "musicgen-medium" in args.models:
        out = wavs_dir / "musicgen-medium"
        out.mkdir(parents=True, exist_ok=True)
        generate_musicgen("facebook/musicgen-medium", metadata, out, device=device)

    if "audioldm2" in args.models:
        out = wavs_dir / "audioldm2"
        out.mkdir(parents=True, exist_ok=True)
        generate_audioldm2("cvssp/audioldm2", metadata, out, device=device)

    if "stable-audio-open" in args.models:
        out = wavs_dir / "stable-audio-open"
        out.mkdir(parents=True, exist_ok=True)
        generate_stable_audio("stabilityai/stable-audio-open-1.0", metadata, out, device=device)

    print("\nListo. Archivos en:", wavs_dir)


if __name__ == "__main__":
    main()
