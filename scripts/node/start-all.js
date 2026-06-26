/**
 * Orquestador Concurrente para levantar Frontend y Backend juntos.
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('=====================================================');
console.log('🎼  Iniciando Sistema de Transformación Musical Híbrida...');
console.log('=====================================================\n');

const backendScript = path.join(__dirname, 'start-backend.js');
const frontendScript = path.join(__dirname, 'start-frontend.js');

const backend = spawn(process.execPath, [backendScript], {
  stdio: 'inherit'
});

const frontend = spawn(process.execPath, [frontendScript], {
  stdio: 'inherit'
});

function cleanupAndExit() {
  console.log('\n[Orquestador] Deteniendo procesos...');
  
  if (backend) {
    console.log('[Orquestador] Deteniendo Backend...');
    backend.kill();
  }
  
  if (frontend) {
    console.log('[Orquestador] Deteniendo Frontend...');
    frontend.kill();
  }
  
  console.log('[Orquestador] Servidores apagados correctamente. ¡Adiós!');
  process.exit(0);
}

// Escuchar señales de interrupción (Ctrl+C)
process.on('SIGINT', cleanupAndExit);
process.on('SIGTERM', cleanupAndExit);

// Si alguno de los procesos falla o se cae
backend.on('close', (code) => {
  if (code !== 0 && code !== null) {
    console.error(`[Orquestador] ✗ El backend finalizó de forma inesperada (código de salida: ${code})`);
    cleanupAndExit();
  }
});

frontend.on('close', (code) => {
  if (code !== 0 && code !== null) {
    console.error(`[Orquestador] ✗ El frontend finalizó de forma inesperada (código de salida: ${code})`);
    cleanupAndExit();
  }
});
