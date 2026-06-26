# Instalacion Limpia

Guia para instalar y ejecutar el proyecto desde cero en macOS y Windows.

## Requisitos

- Python `>=3.11,<3.15`.
- Recomendado si ya esta instalado: Python `3.14.4`.
- FFmpeg para MP3.
- FluidSynth para WAV desde MIDI.
- SoundFont `.sf2` en `assets/soundfonts/default.sf2`.
- Redis solo para modo produccion con Celery.
- CLAP es opcional y pesado; FAD/KLD funcionan sin CLAP.

El frontend es estatico y no requiere build.

## Estructura Esperada

```text
hybrid_engine/
  hybrid_music_engine/
  frontend/
  assets/
    soundfonts/
      default.sf2
  data/
  pyproject.toml
```

No comprimas ni entregues `.venv`.

---

## macOS

### 1. Entrar Al Proyecto

```bash
cd "/Users/brandonpilotzi/Downloads/hybrid_engine"
```

Ajusta la ruta si lo instalas en otra carpeta.

### 2. Dependencias De Sistema

Con Homebrew:

```bash
brew install ffmpeg fluid-synth redis
```

Verificar:

```bash
ffmpeg -version
fluidsynth --version
redis-server --version
```

### 3. Crear Entorno Python

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
```

Si no tienes `python3.14`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
```

### 4. Instalar Dependencias Python

Completa:

```bash
.venv/bin/python -m pip install -e ".[audio,ml,metrics,dev]"
```

Ligera:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

### 5. SoundFont

```bash
mkdir -p assets/soundfonts
test -f assets/soundfonts/default.sf2
```

Si usas otra ruta:

```bash
export HYBRID_SOUNDFONT_PATH="/ruta/a/soundfont.sf2"
```

### 6. Backend En Modo Local

```bash
export HYBRID_ENGINE_ROOT="$(pwd)"
export HYBRID_ENGINE_JOB_BACKEND=local
export HYBRID_ENGINE_REQUIRE_CELERY=0
.venv/bin/python -m uvicorn hybrid_music_engine.api.main:app --reload --host 127.0.0.1 --port 8100
```

API:

```text
http://127.0.0.1:8100/docs
```

### 7. Frontend

En otra terminal:

```bash
cd "/Users/brandonpilotzi/Downloads/hybrid_engine"
python3 -m http.server 5173 --directory frontend
```

Abrir:

```text
http://127.0.0.1:5173
```

### 8. Produccion Con Celery/Redis

Terminal 1:

```bash
redis-server
```

Terminal 2:

```bash
cd "/Users/brandonpilotzi/Downloads/hybrid_engine"
export HYBRID_ENGINE_ROOT="$(pwd)"
export HYBRID_ENGINE_JOB_BACKEND=celery
export HYBRID_ENGINE_REQUIRE_CELERY=1
export HYBRID_ENGINE_BROKER_URL="redis://localhost:6379/0"
export HYBRID_ENGINE_RESULT_BACKEND="redis://localhost:6379/0"
.venv/bin/celery -A hybrid_music_engine.jobs.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

Terminal 3:

```bash
cd "/Users/brandonpilotzi/Downloads/hybrid_engine"
export HYBRID_ENGINE_ROOT="$(pwd)"
export HYBRID_ENGINE_JOB_BACKEND=celery
export HYBRID_ENGINE_REQUIRE_CELERY=1
.venv/bin/python -m uvicorn hybrid_music_engine.api.main:app --reload --host 127.0.0.1 --port 8100
```

Validar Celery:

```bash
.venv/bin/hybrid-engine celery-check
```

---

## Windows PowerShell

### 1. Entrar Al Proyecto

```powershell
cd "C:\Users\Usuario\Desktop\hybrid_engine"
```

### 2. Dependencias De Sistema

FFmpeg con winget:

```powershell
winget install Gyan.FFmpeg
```

Si FFmpeg esta en `C:\Program Files\ffmpeg\bin`:

```powershell
$env:Path += ";C:\Program Files\ffmpeg\bin"
ffmpeg -version
```

FluidSynth:

```powershell
$env:Path += ";C:\Program Files\FluidSynth\bin"
fluidsynth --version
```

Si los ejecutables no estan en PATH, define:

```powershell
$env:HYBRID_FFMPEG_BIN = "C:\Program Files\ffmpeg\bin\ffmpeg.exe"
$env:HYBRID_FLUIDSYNTH_BIN = "C:\Program Files\FluidSynth\bin\fluidsynth.exe"
```

### 3. Crear Entorno Python

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
```

Alternativa:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
```

### 4. Instalar Dependencias Python

Completa:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[audio,ml,metrics,dev]"
```

Ligera:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 5. SoundFont

```powershell
New-Item -ItemType Directory -Force -Path "assets\soundfonts"
Test-Path "assets\soundfonts\default.sf2"
```

Si usas otra ruta:

```powershell
$env:HYBRID_SOUNDFONT_PATH = "C:\ruta\a\soundfont.sf2"
```

### 6. Backend En Modo Local

```powershell
$env:HYBRID_ENGINE_ROOT = (Get-Location).Path
$env:HYBRID_ENGINE_JOB_BACKEND = "local"
$env:HYBRID_ENGINE_REQUIRE_CELERY = "0"
.\.venv\Scripts\python.exe -m uvicorn hybrid_music_engine.api.main:app --reload --host 127.0.0.1 --port 8100
```

### 7. Frontend

En otra terminal:

```powershell
cd "C:\Users\Usuario\Desktop\hybrid_engine"
py -3 -m http.server 5173 --directory frontend
```

Abrir:

```text
http://127.0.0.1:5173
```

### 8. Celery/Redis En Windows

Para pruebas rapidas usa modo local. Para produccion en Windows se recomienda Redis via WSL o Docker.

Ejemplo si Redis esta disponible:

```powershell
$env:HYBRID_ENGINE_ROOT = (Get-Location).Path
$env:HYBRID_ENGINE_JOB_BACKEND = "celery"
$env:HYBRID_ENGINE_REQUIRE_CELERY = "1"
$env:HYBRID_ENGINE_BROKER_URL = "redis://localhost:6379/0"
$env:HYBRID_ENGINE_RESULT_BACKEND = "redis://localhost:6379/0"
.\.venv\Scripts\celery.exe -A hybrid_music_engine.jobs.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

---

## Verificacion Rapida

Backend:

```bash
curl http://127.0.0.1:8100/api/health
```

Frontend:

```text
http://127.0.0.1:5173
```

Validacion estatica:

```bash
node --check frontend/app.js
python3 - <<'PY'
import ast
from pathlib import Path
for path in Path("hybrid_music_engine").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST OK")
PY
```

En Windows:

```powershell
node --check frontend\app.js
.\.venv\Scripts\python.exe -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('hybrid_music_engine').rglob('*.py')]; print('AST OK')"
```

## Evaluacion Minima Recomendada

1. Genera o selecciona al menos una cancion renderizada por genero.
2. En el frontend, abre **Evaluar resultados generados**.
3. Selecciona `1` por genero/corrida.
4. Calcula metricas sin CLAP.
5. Revisa:
   - metricas generales;
   - resumen por genero;
   - tarjetas por cancion con audio generado y original.

## Variables De Entorno

Consulta `.env.example`. En PowerShell define variables con:

```powershell
$env:NOMBRE_VARIABLE = "valor"
```

En macOS/Linux:

```bash
export NOMBRE_VARIABLE="valor"
```
