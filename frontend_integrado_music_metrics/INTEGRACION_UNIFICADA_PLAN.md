# Plan de integración unificada del frontend

## Diagnóstico

El proyecto contiene dos líneas de frontend: una línea estática funcional formada por `index_modified.html`, `styles.css`, `styles_demo_additions.css`, `app.js` y `demo.js`; y una línea de scaffold React/Vite indicada por `package.json`, `vite.config.ts`, `index.css` y `Hero.tsx`. En la copia actual no existe la carpeta `client/src` que espera Vite, por lo que el `index.html` activo apunta a `/src/main.tsx` y no puede levantar la interfaz real sin reconstruir esa estructura.

## Decisión de integración

La integración se realizará sobre la línea estática existente porque es la que contiene el motor completo de transformación musical y el demo documentado. El objetivo inmediato es que `index.html` sea el punto único de entrada y cargue una interfaz consolidada donde convivan: el motor principal, el menú de navegación, el submenu de demo y las secciones de gráficas, audios y documentación.

## Cambios propuestos

| Área | Acción | Resultado esperado |
|---|---|---|
| Punto de entrada | Reemplazar `index.html` con la versión integrada basada en `index_modified.html` | La app deja de depender de `/src/main.tsx` ausente |
| Estilos | Incorporar `styles_demo_additions.css` dentro de `styles.css` | Un solo CSS principal, sin pérdida del modal |
| Navegación | Añadir un menú principal con anclas internas | Convivencia entre motor, demo, gráficas, audios y documentación |
| Demo | Mantener `demo.js` como módulo aislado | Apertura/cierre del modal y tabs sin interferir con `app.js` |
| Compatibilidad | Conservar IDs usados por `app.js` | El flujo del motor musical mantiene su comportamiento |

## Criterio de éxito

La integración será exitosa si el navegador abre `index.html`, muestra el menú unificado, conserva el flujo del motor, abre el demo con sus cuatro pestañas, no reporta errores por elementos faltantes críticos y permite navegar a gráficas, audios y documentación desde el mismo frontend.
