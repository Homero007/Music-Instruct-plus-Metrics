# 🚀 Hybrid Engine en Google Colab

Ejecuta entrenamientos del Transformer + pruebas de métricas en la nube de forma GRATUITA.

## ⚡ Quick Start (3 pasos)

### 1️⃣ Prepara datos en Google Drive
```bash
# En tu máquina local:
tar -czf jamendo_150.tar.gz data/datasets/jamendo/delivery_jamendo_150/

# Sube jamendo_150.tar.gz a Google Drive en una carpeta llamada "hybrid_engine"
```

### 2️⃣ Abre el notebook Colab
[**→ Abrir notebook en Google Colab**](https://colab.research.google.com/notebook)

(Aún no tenemos URL directa; copiar `hybrid_engine_colab.ipynb` a tu Google Drive y abrir desde ahí)

### 3️⃣ Ejecuta las celdas en orden
- Setup (instala dependencias, ~2 min)
- Monta Google Drive (requiere autorización)
- Descarga datos (extrae del Drive, ~2-3 min)
- Entrena (tarda según epochs)
- Guarda resultados en Drive

---

## 📋 Opciones de ejecución

### A) **Notebook interactivo** (recomendado para experimentos)
**Archivo:** `hybrid_engine_colab.ipynb`

**Ventajas:**
- Visualización en tiempo real
- Fácil de entender (celdas paso a paso)
- Puedes pausar/reanudar

**Desventajas:**
- Manual (tienes que pulsar "run" cada celda)
- Se desconecta si inactivo >12h

### B) **Script CLI** (recomendado para entrenamientos largos)
**Archivo:** `train_colab.py`

**Uso en Colab:**
```python
!python train_colab.py --epochs 100 --batch-size 8 --save-every 10
```

**Ventajas:**
- Totalmente automatizado
- Fácil reproducible
- Checkpoints automáticos

**Desventajas:**
- Menos interactivo
- Requiere menos supervisión (perfecto para dejar corriendo)

---

## 🔧 Instalación manual (si no usas el notebook)

```python
# En una celda de Colab:
!pip install -q torch transformers numpy scipy pandas librosa demucs \
  matplotlib openpyxl miditok pretty_midi

# Clona el repo (opcional, para código completo):
!git clone https://github.com/tu_usuario/hybrid_engine.git /content/hybrid_engine
```

---

## 📊 Entrenamiento paso a paso

### Configuración default
```python
BATCH_SIZE = 8          # Ajusta a 4 si se queda sin RAM
LEARNING_RATE = 1e-4
EPOCHS = 100           # Reduce a 50 para pruebas rápidas
CHECKPOINT_EVERY = 10  # Guarda cada 10 epochs
```

### Monitoreo
- **Loss convergiendo:** ✅ espera, es normal
- **CUDA out of memory:** reduce `BATCH_SIZE` a 4 o 2
- **Desconexión por inactividad:** aumenta `CHECKPOINT_EVERY` para checkpoints más frecuentes

---

## 💾 Datos y Checkpoints

### Estructura en Google Drive
```
MyDrive/
└── hybrid_engine/
    ├── jamendo_150.tar.gz          (sube tú)
    ├── checkpoints/                 (crea automáticamente)
    │   ├── model_ep010.pt
    │   ├── model_ep020.pt
    │   └── ...
    └── results/                     (crea automáticamente)
        ├── training_metrics.json
        └── summary.json
```

### Recuperar entrenamiento interrumpido
```python
# Carga último checkpoint y continúa
checkpoint = torch.load('/content/drive/MyDrive/hybrid_engine/checkpoints/model_ep050.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## ⚠️ Limitaciones de Colab

| Límite | Valor | Workaround |
|--------|-------|-----------|
| **Tiempo sesión** | 12h inactividad | Checkpoint cada 30min |
| **RAM** | 12 GB | Batch size ≤ 8 |
| **GPU VRAM** | ~16 GB (T4) | Mixed precision + gradient checkpointing |
| **Almacenamiento temp** | 100 GB | Guardar en Drive, no acumular |
| **Tiempo total uso GPU** | ~40h/semana | Usar Kaggle si necesitas más |

---

## 🔄 Alternativas a Colab

### 🥇 **Kaggle Notebooks** (mejor para entrenamientos largos)
- **GPU:** P100 (más fuerte que T4)
- **Tiempo:** 30h/semana gratis
- **Almacenamiento:** 500 GB/mes
- **Ventaja:** Mejor para modelos grandes

**Uso:** 
```python
# Mismo notebook, funciona igual
```

### 🥈 **Hugging Face Spaces** (para demos)
- **CPU solo** (gratis) o **GPU rentado** (~$1/h)
- **Deploy automático:** push a HF Hub → app viva
- **Ideal para:** interfaces Gradio, compartir modelos

### 🥉 **Colab Pro** (pagado)
- $10/mes
- GPU T4 más rápida, TPU disponible
- Sesiones hasta 24h
- Vale si entrenas regularmente

---

## 🎯 Ejemplo: Entrenamiento end-to-end

```python
# 1. Setup (ejecutar una vez)
# (instala + monta Drive + descarga datos)

# 2. Personaliza config
EPOCHS = 50
BATCH_SIZE = 8

# 3. Entrena
for epoch in range(EPOCHS):
    loss = train_epoch(...)
    if epoch % 10 == 0:
        save_checkpoint(...)

# 4. Exporta
# (CSV + JSON → Google Drive)

# 5. Descarga desde Drive y analiza localmente
# (métricas.csv, checkpoints para usar después)
```

---

## 📚 Siguiente: Integración con tu pipeline local

Una vez que tengas checkpoints en Drive:
1. Descárgalos
2. Cópialo a `data/models/tokens/`
3. Úsalos en tu pipeline local o frontend
4. Genera audios, métricas, etc.

```bash
# Ejemplo: usar modelo de Colab localmente
cp ~/Downloads/model_ep050.pt \
   data/models/tokens/jamendo_transformer_colab/checkpoint.pt

./start.sh  # frontend local + backend
```

---

## ❓ FAQ

**P: ¿Puedo entrenar modelos reales (no sintéticos) en Colab?**  
R: Sí, pero tardará más. Con datos reales de Jamendo (15 clips) + 50 epochs ≈ 30 min en T4.

**P: ¿Se pierden los checkpoints si Colab se desconecta?**  
R: No, están en Google Drive. Carga el último y continúa.

**P: ¿Puedo usar múltiples GPUs en Colab?**  
R: Solo hay una GPU T4. Para multi-GPU, usa GCP, AWS o Lambda Labs ($$$).

**P: ¿Cómo integro métricas reales (CLAP, PaSST) en Colab?**  
R: Instala `pip install laion-clap hear21passt torchvggish panns_inference` en setup.

---

## 📞 Soporte

- **Error de CUDA:** reduce batch size
- **Out of Memory:** reinicia sesión, reduce epochs o num_samples
- **Desconexión:** aumenta frecuencia de checkpoints
- **Datos no encontrados:** verifica estructura en Drive

---

**¡A entrenar! 🎉**
