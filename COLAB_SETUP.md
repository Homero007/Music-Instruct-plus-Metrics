# Migración a Google Colab — Guía Completa

## Requisitos previos
- Cuenta de Google (para Colab + Drive)
- Datos de Jamendo comprimidos (~10-15GB en Drive, ideal .tar.gz)
- Modelos preentrenados (descargables on-demand o en Drive)

## Paso 1: Preparar datos en Google Drive

### Opción A: Comprimir localmente (recomendado)
```bash
cd /Users/homer/Downloads/hybrid_engine
tar -czf jamendo_150.tar.gz data/datasets/jamendo/delivery_jamendo_150/
# Sube jamendo_150.tar.gz a tu Google Drive en una carpeta llamada "hybrid_engine"
```

### Opción B: Usar Hugging Face Hub (alternativa)
```bash
huggingface-cli login
huggingface-cli upload homero/hybrid-engine-data jamendo_150.tar.gz
```

## Paso 2: Crear notebook en Colab

### Enlace directo a notebook template
Copia este notebook: **[Colab Template - Hybrid Engine](https://colab.research.google.com/drive/your_notebook_id)**

(Lo crearemos abajo)

## Paso 3: Estructura del notebook

El notebook tiene 6 secciones:

### 1️⃣ **Setup inicial (ejecutar una sola vez)**
- Instala dependencias
- Monta Google Drive
- Descarga/extrae datos

### 2️⃣ **Importaciones y configuración**
- Imports del proyecto
- Config de CUDA/PyTorch

### 3️⃣ **Carga de datos**
- Extrae Jamendo de Drive
- Indexa clips

### 4️⃣ **Entrenamiento (Transformer)**
- Loop de entrenamiento
- Checkpoint cada N epochs
- Guarda modelos en Drive

### 5️⃣ **Pruebas rápidas**
- Test KAD, CLAP, KLD
- Generación de sample

### 6️⃣ **Exportación**
- Descarga checkpoints
- Genera reportes CSV

## Limitaciones de Colab a tener en cuenta

| Recurso | Límite | Workaround |
|---------|--------|-----------|
| **Tiempo sesión** | 12h inactividad → desconexión | Guardar checkpoints cada 30min |
| **RAM** | 12GB | Batch size 8-16 |
| **GPU VRAM** | ~16GB (T4) | Gradient checkpointing, mixed precision |
| **Almacenamiento** | 100GB temporal | Guardar en Drive, no acumular en /tmp |
| **Ancho de banda** | No limitado (upstream lento) | Pre-comprimir datos |

## Paso 4: Checkpoints y recuperación

### Auto-save cada 30 minutos
```python
def save_checkpoint(model, optimizer, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

# En el loop de entrenamiento:
if epoch % 30 == 0:  # cada 30 epochs
    save_checkpoint(model, opt, epoch, loss, 
                    '/content/drive/MyDrive/hybrid_engine/checkpoints/model_ep{epoch}.pt')
```

### Reanudar entrenamiento
```python
checkpoint = torch.load('/content/drive/MyDrive/hybrid_engine/checkpoints/model_ep200.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Paso 5: Alternativas más rápidas

Si el entrenamiento es muy lento:
- **Kaggle Notebooks** (GPU P100 gratis, 30h por semana)
- **HuggingFace Spaces + paid compute** (GPU A100 rentado)
- **Vast.ai o Lambda Labs** (GPU on-demand barata, ~$0.30/h)

## Siguiente: copiar el notebook template abajo
