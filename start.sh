#!/bin/bash

# Configuration
export HYBRID_ENGINE_ROOT="$(pwd)"
export HYBRID_ENGINE_JOB_BACKEND="local"
export HYBRID_ENGINE_REQUIRE_CELERY="0"
export HYBRID_SOUNDFONT_PATH="$HYBRID_ENGINE_ROOT/assets/soundfonts/default.sf2"

# Resolve binaries
export HYBRID_FLUIDSYNTH_BIN="$(which fluidsynth 2>/dev/null || echo 'fluidsynth')"
export HYBRID_FFMPEG_BIN="$(which ffmpeg 2>/dev/null || echo 'ffmpeg')"

echo "====================================================="
echo "🎼  Iniciando Sistema de Transformación Musical Híbrida..."
echo "====================================================="
echo "Project Root: $HYBRID_ENGINE_ROOT"
echo "SoundFont:    $HYBRID_SOUNDFONT_PATH"
echo "FluidSynth:   $HYBRID_FLUIDSYNTH_BIN"
echo "FFmpeg:       $HYBRID_FFMPEG_BIN"
echo "====================================================="

# Check requirements
if [ ! -f "$HYBRID_SOUNDFONT_PATH" ]; then
    echo "⚠️  Advertencia: No se encontró la SoundFont en $HYBRID_SOUNDFONT_PATH"
fi

# Determine Python virtualenv
if [ -d ".venv-mac" ]; then
    PYTHON_BIN=".venv-mac/bin/python"
elif [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# ── Modo entrenamiento ───────────────────────────────────────────────
# Uso:
#   ./start.sh            -> arranca backend + frontend (por defecto)
#   ./start.sh --train    -> entrena el Transformer (200 epochs) y termina
if [ "$1" = "--train" ] || [ "$1" = "train" ]; then
    echo "🧠 Modo entrenamiento: Transformer (200 epochs)..."
    echo "====================================================="
    # Activa el venv (igual que el flujo de Colab) con fallback a PYTHON_BIN
    if [ -f ".venv-mac/bin/activate" ]; then
        source .venv-mac/bin/activate
        TRAIN_PY="python3"
    else
        TRAIN_PY="$PYTHON_BIN"
    fi
    $TRAIN_PY train_colab.py \
        --epochs 200 \
        --batch-size 8 \
        --save-every 10 \
        --checkpoint-dir ./checkpoints/transformer_200ep \
        --results-dir ./results/transformer_200ep
    STATUS=$?
    echo "====================================================="
    if [ $STATUS -eq 0 ]; then
        echo "✅ Entrenamiento terminado. Checkpoints en ./checkpoints/transformer_200ep"
    else
        echo "❌ El entrenamiento falló (código $STATUS). Revisa la salida de arriba."
    fi
    exit $STATUS
fi

echo "Usando Python de: $PYTHON_BIN"
echo "====================================================="

# Start Backend
echo "🚀 Iniciando Backend (FastAPI) en http://127.0.0.1:8100..."
$PYTHON_BIN -m uvicorn hybrid_music_engine.api.main:app --host 127.0.0.1 --port 8100 > backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "🌐 Iniciando Frontend en http://127.0.0.1:5173..."
python3 -m http.server 5173 --directory frontend > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait a brief moment to ensure startup
sleep 1.5

# Check if processes are running
if ps -p $BACKEND_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ ¡Ambos servicios se están ejecutando!"
    echo "   👉 Frontend: http://127.0.0.1:5173"
    echo "   👉 Backend (API Docs): http://127.0.0.1:8100/docs"
    echo "-----------------------------------------------------"
    echo "Presiona [Ctrl+C] para detener todos los servicios."
    echo "-----------------------------------------------------"
else
    echo "❌ Error al iniciar alguno de los servicios."
    if ! ps -p $BACKEND_PID > /dev/null; then
        echo "   El backend falló al iniciar. Revisa backend.log"
    fi
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo "   El frontend falló al iniciar. Revisa frontend.log"
    fi
fi

cleanup() {
    echo -e "\n🛑 Deteniendo servicios..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servicios detenidos correctamente. ¡Hasta luego!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
wait
