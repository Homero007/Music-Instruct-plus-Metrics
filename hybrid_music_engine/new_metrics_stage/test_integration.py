"""Verifica el disparo automático new_metrics.run_after_render (post-render)."""
from __future__ import annotations
import os, tempfile
from pathlib import Path
from types import SimpleNamespace
import matplotlib; matplotlib.use("Agg")
import numpy as np, soundfile as sf
from new_metrics.integration import run_after_render

def _tone(path, bpm, hz, sr=22050, sec=4.0, noise=0.0, seed=0):
    rng=np.random.default_rng(seed); n=int(sr*sec); t=np.arange(n)/sr
    y=0.05*np.sin(2*np.pi*hz*t)
    for k in range(int(sec/(60/bpm))):
        i=int(k*(60/bpm)*sr)
        if i<n:
            e=np.exp(-np.arange(min(800,n-i))/60); y[i:i+len(e)]+=e*0.6
    if noise: y+=noise*rng.standard_normal(n)
    y=0.9*y/(np.max(np.abs(y))+1e-9); path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr)

def main():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); data=root/"data"; run_id="ranked_demo_001"
        # renders/<run_id>/<candidate>/<cand>.wav (estructura real de ranked.py)
        for c in range(1,4):
            _tone(data/"renders"/run_id/str(c)/f"{c}.wav", 110+c, 330, noise=0.03, seed=c)
        # referencia real para FAD
        real=root/"real"
        for i in range(3): _tone(real/f"r_{i}.wav", 112, 330, seed=100+i)
        os.environ["HYBRID_NEW_METRICS_REAL"]=str(real)
        os.environ["HYBRID_NEW_METRICS_AUTO"]="1"
        os.environ.pop("HYBRID_T5_DIR", None)

        cfg=SimpleNamespace(data_dir=data, project_root=root)
        report=run_after_render(cfg, run_id, condition_genre="electronic")
        assert report is not None, "no corrió"
        out=data/"new_metrics"/run_id
        assert (out/"report.json").exists()
        assert len(report["plots"])>=1, report["plots"]
        # No escribió ranking (separado del reward model)
        assert not list(out.rglob("ranking*.json"))
        print("ok run_after_render  ->", out)
        print("   métricas:", report["audio"].get("ranking", "n/a") and "audio OK")

        # AUTO=0 desactiva
        os.environ["HYBRID_NEW_METRICS_AUTO"]="0"
        assert run_after_render(cfg, run_id) is None
        print("ok toggle HYBRID_NEW_METRICS_AUTO=0 desactiva")
    print("TODAS LAS PRUEBAS DE INTEGRACIÓN PASARON")

if __name__=="__main__":
    main()
