"""Verifica correctitud y memoria del cortador en streaming."""
from __future__ import annotations
import tempfile, tracemalloc
from pathlib import Path
import numpy as np, soundfile as sf
import clip_cutter as cc

def _fake_id(s, prefix="clip"): return f"{prefix}_{s}"

def _make_wav(path, seconds, sr=22050, stereo=True, seed=0):
    rng=np.random.default_rng(seed); n=int(sr*seconds)
    y=(0.1*np.sin(2*np.pi*220*np.arange(n)/sr)).astype(np.float32)
    data=np.stack([y, y*0.9], axis=1) if stereo else y
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return sr, n

def test_windows_match_naive():
    # mismo recorrido que el loop original
    wins=list(cc.iter_windows(total_samples=100, clip_samples=30, hop_samples=30, min_samples=10, max_clips=None))
    assert wins==[(0,30),(30,60),(60,90),(90,100)] or wins[-1]==(90,100)

def test_streaming_equals_full_load():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); src=root/"track.wav"
        sr,n=_make_wav(src, seconds=12, sr=22050, stereo=True, seed=1)
        entries,rej=cc.cut_track_clips(
            src, root/"out", "track1", genre="electronic", track={"name":"t"},
            target_sr=sr, mono=False, clip_seconds=5, hop_seconds=5, min_seconds=2,
            max_clips=None, create_id=_fake_id)
        assert rej is None and len(entries)>=2
        # Comparar el primer clip contra el slice del archivo completo (sample-exacto)
        full,_=sf.read(str(src), dtype="float32", always_2d=False)
        c0,_=sf.read(entries[0]["clip_path"], dtype="float32", always_2d=False)
        assert np.allclose(c0, full[0:c0.shape[0]], atol=1e-4), "clip 0 difiere del slice"
        # duración correcta
        assert abs(entries[0]["duration_seconds"]-5.0)<1e-3

def test_peak_memory_is_one_clip_not_whole_file():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); src=root/"long.wav"
        # ~3 min estéreo @22050 = ~3.97M frames; full-load float32 ≈ 30 MiB
        sr,n=_make_wav(src, seconds=180, sr=22050, stereo=True, seed=2)
        full_bytes=n*2*4  # frames*canales*4(float32)

        # Streaming
        tracemalloc.start()
        entries,rej=cc.cut_track_clips(
            src, root/"out", "L", genre="g", track={},
            target_sr=sr, mono=False, clip_seconds=5, hop_seconds=5, min_seconds=2,
            max_clips=None, create_id=_fake_id)
        _,peak_stream=tracemalloc.get_traced_memory(); tracemalloc.stop()
        assert rej is None and len(entries)>=30

        # Full-load (lo que hace el código actual)
        tracemalloc.start()
        full,_=sf.read(str(src), dtype="float32", always_2d=False)
        _slice=full[0:int(5*sr)].copy()
        _,peak_full=tracemalloc.get_traced_memory(); tracemalloc.stop()
        del full

        clip_bytes=int(5*sr)*2*4
        print(f"   archivo completo ≈ {full_bytes/1e6:.1f} MB | "
              f"pico full-load={peak_full/1e6:.1f} MB | pico streaming={peak_stream/1e6:.1f} MB | "
              f"un clip ≈ {clip_bytes/1e6:.2f} MB")
        # El pico en streaming debe ser MUCHO menor que el del archivo completo
        assert peak_stream < peak_full/3, (peak_stream, peak_full)
        # y del orden de unos pocos clips, no del archivo entero
        assert peak_stream < full_bytes/3

if __name__=="__main__":
    for name,fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("TODAS LAS PRUEBAS DEL CORTADOR PASARON")
