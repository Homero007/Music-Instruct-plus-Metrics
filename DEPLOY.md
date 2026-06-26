# Desplegar el frontend (Netlify / Vercel)

El frontend (`frontend/`) es 100% estático (HTML/CSS/JS, sin build). Se puede
publicar tal cual. La parte pesada (backend) sigue corriendo aparte: en tu
máquina o en la GPU de la nube (Colab).

## Arquitectura tras el deploy

```
Sitio público (Netlify/Vercel, https)
   │
   ├── URL del backend (header)  ──►  Backend completo (FastAPI)
   │                                   local 127.0.0.1:8100  o  tu servidor
   │                                   (proyectos, import, entrenamiento, render…)
   │
   └── Backend en la nube (Paso 4) ──► Microservicio GPU en Colab (túnel https)
                                        SOLO generación + métricas (FAD/CLAP/KLD/KAD)
```

El backend se resuelve **automáticamente** (sin escribir URLs en la UI):

1. **`?backend=<url>`** o **`?cloud=<url>`** en la dirección → se guardan solos
   (enlace "1-click"; el notebook de Colab imprime uno con `?cloud=<túnel>`).
2. Si abriste el sitio en **localhost** → backend local `http://127.0.0.1:8100`.
3. Si está **desplegado** → llama a su propio origen `/api/*` y **Netlify hace de
   proxy** hacia el backend real (configurado en `netlify.toml`). Sin CORS ni
   contenido mixto.

El **Backend en la nube (GPU)** del Paso 4 (túnel de Colab para generar/medir) es
independiente y también se autoconfigura con `?cloud=<url>`.

## Opción A · Netlify (arrastrar y soltar)

1. Entra a <https://app.netlify.com/drop>.
2. Arrastra la carpeta **`frontend/`** (no el repo entero).
3. Te da un link tipo `https://mi-sitio.netlify.app`.
4. **Contraseña (gratis):** `Site settings > Access control > Password protection > Set password`.

Deploy por Git: conecta el repositorio; Netlify lee `netlify.toml` (publica `frontend/`).

## Opción B · Vercel

1. `Add New > Project` e importa el repo.
2. En **Root Directory** elige `frontend` (o deja que lea `vercel.json`, que ya
   apunta `outputDirectory` a `frontend`).
3. Framework Preset: **Other**. Sin build.
4. Te da un link `https://mi-sitio.vercel.app`.
5. Contraseña: requiere plan Pro (Deployment Protection). Para contraseña gratis, usa Netlify.

## Configurar el backend desde el sitio público

1. **Backend completo (proxy de Netlify):** en `netlify.toml`, cambia `BACKEND_URL`
   por la dirección pública de tu backend (debe ser accesible desde internet; un
   `127.0.0.1` no sirve porque el proxy corre en los servidores de Netlify) y vuelve
   a desplegar. El frontend ya llama a `/api/*` sin que tú escribas nada.
2. **Backend GPU (Colab):** abre el sitio con el enlace que imprime el notebook,
   `https://TU-SITIO.netlify.app/?cloud=https://xxxx.trycloudflare.com`, o ve al
   **Paso 4** y pulsa **Conectar**.

## Avisos importantes

- **Sin contenido mixto** gracias al proxy: el navegador solo habla con tu dominio
  https de Netlify, y Netlify reenvía al backend desde su servidor. Esto también
  evita el bloqueo de Safari.
- **CORS ya está abierto** en el backend (`allow_origins=["*"]`); útil si en vez del
  proxy prefieres apuntar con `?backend=<url>` directamente.
- El backend completo **no** está pensado para exponerse públicamente sin protección;
  para una demo, lo natural es backend local + sitio con contraseña.
