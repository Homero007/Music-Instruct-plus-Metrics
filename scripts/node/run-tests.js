/**
 * Script para ejecutar las pruebas unitarias
 * Corre pytest de forma segura en Windows y macOS.
 */

const { spawn } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

const isWin = os.platform() === 'win32';
const venvPython = isWin 
  ? path.join('.venv', 'Scripts', 'python.exe')
  : path.join('.venv', 'bin', 'python');

const projectRoot = path.resolve(__dirname, '..', '..');
const pythonPath = path.resolve(projectRoot, venvPython);

if (!fs.existsSync(pythonPath)) {
  console.error(`✗ Error: No se encontró Python en: ${pythonPath}`);
  process.exit(1);
}

console.log('[Tests] Iniciando pytest a través del entorno virtual...');

const child = spawn(pythonPath, ['-m', 'pytest'], {
  cwd: projectRoot,
  stdio: 'inherit'
});

child.on('close', (code) => {
  process.exit(code || 0);
});
