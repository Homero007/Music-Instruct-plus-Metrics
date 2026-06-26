/**
 * Script de Autodiagnóstico del Entorno
 * Comprueba dependencias de sistema y archivos requeridos.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

console.log('=====================================================');
console.log('🔍  Comprobación de Requisitos y Diagnóstico del Entorno');
console.log('=====================================================\n');

let allOk = true;
const isWin = os.platform() === 'win32';

// 1. Entorno Virtual Python
const venvPath = path.resolve(__dirname, '..', '..', '.venv');
const pythonExec = isWin 
  ? path.join(venvPath, 'Scripts', 'python.exe')
  : path.join(venvPath, 'bin', 'python');

if (fs.existsSync(venvPath) && fs.existsSync(pythonExec)) {
  console.log('✅ [VENV] Entorno virtual Python encontrado correctamente.');
} else {
  console.log('❌ [VENV] Entorno virtual (.venv) no encontrado o incompleto en: ' + venvPath);
  console.log('   Ejecuta: python -m venv .venv');
  allOk = false;
}

// 2. FFmpeg
try {
  const ffmpegVer = execSync('ffmpeg -version', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
  const firstLine = ffmpegVer.split('\n')[0];
  console.log(`✅ [FFMPEG] FFmpeg disponible: "${firstLine.trim()}"`);
} catch (e) {
  console.log('❌ [FFMPEG] FFmpeg no está disponible en el PATH del sistema.');
  console.log('   Windows winget: winget install Gyan.FFmpeg');
  console.log('   macOS Homebrew: brew install ffmpeg');
  allOk = false;
}

// 3. FluidSynth
try {
  const fluidVer = execSync('fluidsynth --version', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
  const firstLine = fluidVer.split('\n')[0];
  console.log(`✅ [FLUIDSYNTH] FluidSynth disponible: "${firstLine.trim()}"`);
} catch (e) {
  console.log('❌ [FLUIDSYNTH] FluidSynth no está disponible en el PATH.');
  console.log('   Windows: Descarga el zip oficial y añade la carpeta bin al PATH.');
  console.log('   macOS Homebrew: brew install fluid-synth');
  allOk = false;
}

// 4. SoundFont
const sfPath = path.resolve(__dirname, '..', '..', 'assets', 'soundfonts', 'default.sf2');
if (fs.existsSync(sfPath)) {
  console.log(`✅ [SOUNDFONT] Archivo SoundFont encontrado en: ${sfPath}`);
} else {
  console.log(`⚠️  [SOUNDFONT] Archivo default.sf2 no encontrado en: ${sfPath}`);
  console.log('   El renderizado de MIDI a WAV requiere colocar una SoundFont (.sf2) en esa ruta.');
  console.log('   Puedes descargar una (ej. FluidR3_GM.sf2) y renombrarla.');
}

// 5. Dependencias Python del Motor
if (fs.existsSync(pythonExec)) {
  try {
    const pkgCheck = execSync(`"${pythonExec}" -c "import hybrid_music_engine; print('ok')"`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
    if (pkgCheck.trim() === 'ok') {
      console.log('✅ [PYTHON PKG] Librería "hybrid_music_engine" instalada en modo editable.');
    } else {
      throw new Error();
    }
  } catch (e) {
    console.log('❌ [PYTHON PKG] La librería "hybrid_music_engine" no está instalada en el venv.');
    console.log('   Ejecuta: .\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]"');
    allOk = false;
  }
}

console.log('\n=====================================================');
if (allOk) {
  console.log('🎉 ¡El entorno local está listo para ejecutar la prueba!');
} else {
  console.log('⚠️  Hay requisitos pendientes. Por favor resuélvelos.');
}
console.log('=====================================================\n');
process.exit(allOk ? 0 : 1);
