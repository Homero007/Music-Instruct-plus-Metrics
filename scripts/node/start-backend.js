/**
 * Script de inicio para el backend de FastAPI
 * Detecta el sistema operativo, configura las variables de entorno locales y corre Uvicorn.
 */

const { spawn } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

const isWin = os.platform() === 'win32';
const venvPython = isWin 
  ? path.join('.venv', 'Scripts', 'python.exe')
  : path.join('.venv', 'bin', 'python');

const projectRoot = path.join(__dirname, '..', '..');
const pythonPath = path.resolve(projectRoot, venvPython);

if (!fs.existsSync(pythonPath)) {
  console.error(`✗ Error: No se encontró el ejecutable de Python en el venv: ${pythonPath}`);
  console.error('  Por favor asegúrate de haber creado el entorno virtual con: python -m venv .venv');
  process.exit(1);
}

// Configuración de variables de entorno para modo local
const env = {
  ...process.env,
  HYBRID_ENGINE_JOB_BACKEND: 'local',
  HYBRID_ENGINE_REQUIRE_CELERY: '0'
};

console.log(`[Backend] Iniciando Uvicorn con Python venv: ${venvPython}...`);

const args = ['-m', 'uvicorn', 'hybrid_music_engine.api.main:app', '--reload', '--host', '127.0.0.1', '--port', '8100'];

const child = spawn(pythonPath, args, {
  cwd: projectRoot,
  stdio: 'inherit',
  env
});

child.on('error', (err) => {
  console.error('[Backend] Error al iniciar el subproceso:', err);
  process.exit(1);
});

child.on('close', (code) => {
  console.log(`[Backend] Proceso finalizado con código: ${code}`);
  process.exit(code || 0);
});
