/**
 * Servidor estático nativo de Node.js (Sin dependencias externas)
 * Sirve los archivos de la carpeta 'frontend' en el puerto 5173.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5173;
const HOST = '127.0.0.1';
const PUBLIC_DIR = path.resolve(__dirname, '..', '..', 'frontend');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.pdf': 'application/pdf'
};

const server = http.createServer((req, res) => {
  // Manejo de la URL decoded para soportar espacios/acentos en el filesystem
  const decodedUrl = decodeURIComponent(req.url);
  
  // Limpiar query params y fragmentos de la ruta
  const cleanUrl = decodedUrl.split('?')[0].split('#')[0];
  
  let filePath = path.join(PUBLIC_DIR, cleanUrl === '/' ? 'index.html' : cleanUrl);
  
  // Seguridad básica contra Directory Traversal
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.statusCode = 403;
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.end('403 Acceso Prohibido');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      // Si el archivo no existe, retornar 404
      res.statusCode = 404;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('404 Archivo No Encontrado');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    res.writeHead(200, {
      'Content-Type': contentType,
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff'
    });

    const stream = fs.createReadStream(filePath);
    stream.on('error', (streamErr) => {
      console.error('[Frontend Server] Error de lectura de archivo:', streamErr);
      if (!res.headersSent) {
        res.statusCode = 500;
        res.end('500 Error Interno del Servidor');
      }
    });
    stream.pipe(res);
  });
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`✗ Error: El puerto ${PORT} ya está en uso. ¿Tienes otro servidor corriendo?`);
  } else {
    console.error('[Frontend Server] Error:', err);
  }
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  console.log(`[Frontend] Servidor web estático en funcionamiento:`);
  console.log(`           👉  http://${HOST}:${PORT}`);
  console.log(`[Frontend] Sirviendo archivos desde: ${PUBLIC_DIR}`);
});
