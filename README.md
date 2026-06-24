# TFE — Diagnóstico de la Malla Vial en Bogotá D.C.

## Mediante Visión por Computadora y Redes Neuronales Convolucionales

**Universidad Internacional de La Rioja (UNIR) — Máster en Inteligencia Artificial 2026**

**Autores:** Andrés Felipe Parra Forero · Marelis del Carmen Díaz Genes · María Jaqueline Velásquez Parrado  
**Directora:** María Inmaculada Mohino Herranz

---

## Descripción

Sistema de detección automática de fallas viales en Bogotá D.C. basado en **YOLOv8n + Claude AI (Anthropic)**. Detecta y clasifica cuatro tipos de fallas en pavimento flexible urbano:

| Clase | Color en interfaz | mAP@50 |
|-------|------------------|--------|
| Hueco | Azul | 49.13% |
| Piel de Cocodrilo | Azul claro | 66.09% |
| Grieta | Naranja | 56.53% |
| Parcheo | Verde | 72.73% |

**mAP@50 global: 59.80%**

El sistema aplica reglas de negocio **ASTM D6433 / SIGMA UMV** para determinar la severidad (BAJA / MEDIA / ALTA) y la intervención recomendada (SI / SF / PA / BA) para cada falla detectada.

---

## Arquitectura del Sistema

```
Imagen de entrada
       ↓
   YOLOv8n
  (Detección)
       ↓
Reglas ASTM D6433     →    Severidad + Intervención
       ↓
  Claude AI API
 (Análisis profundo)   →    Causa · Severidad 1-5 · Acción · Tiempo
       ↓
  Reporte PDF
 (Descarga automática)
```

---

## Archivos del Proyecto

| Archivo | Descripción |
|---------|-------------|
| `app_huecos.py` | Interfaz Gradio con pipeline YOLOv8 + Claude AI + exportación PDF |
| `entrenar_yolo.py` | Entrenamiento YOLOv8n con dataset combinado (corpus bogotano + RDD2022) |
| `evaluar_modelo.py` | Evaluación del modelo y exportación de métricas a CSV |
| `descargar_umv.py` | Descarga de imágenes del repositorio UMV |
| `logo-umv-bogota.webp` | Logo institucional UAERMV Bogotá |

---

## Dataset

| Dataset | Imágenes | Descripción |
|---------|----------|-------------|
| Corpus Bogotá | 646 | Imágenes propias capturadas por Ingenieros Diagnosticadores de la UAERMV |
| RDD2022 | ~25.374 | Dataset internacional de daños viales (Road Damage Dataset 2022) |
| **Total combinado** | **26.020** | Dataset final de entrenamiento |

División: 88% entrenamiento / 8% validación / 4% prueba

---

## Modelo Entrenado

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | YOLOv8n (Ultralytics) |
| Pesos iniciales | COCO pretrained (yolov8n.pt) |
| Épocas | 100 |
| Batch size | 16 |
| Tasa de aprendizaje | 0.01 (AdamW) |
| Tamaño de imagen | 640 × 640 px |
| Hardware | NVIDIA RTX 4070 |
| Tiempo de entrenamiento | ~9 horas |
| Umbral de confianza (demo) | 0.05 |

> **Nota:** El archivo `best.pt` (modelo entrenado) no está incluido en este repositorio por su tamaño. Puede solicitarse a los autores o reentrenarse con los scripts disponibles.

---

## Requisitos

```bash
pip install gradio ultralytics anthropic opencv-python fpdf2 numpy
```

| Requisito | Versión |
|-----------|---------|
| Python | 3.10 o superior |
| gradio | 4.x o superior |
| ultralytics | 8.x |
| anthropic | última disponible |
| fpdf2 | última disponible |

---

## Uso

### 1. Configurar la API Key de Anthropic

Editar la línea 13 de `app_huecos.py` con la clave obtenida en [console.anthropic.com](https://console.anthropic.com/settings/keys):

```python
ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

### 2. Verificar la ruta del modelo

Editar la línea 14 de `app_huecos.py` con la ruta local al archivo `best.pt`:

```python
modelo = YOLO(r"C:\ruta\al\modelo\best.pt")
```

### 3. Iniciar la aplicación

```bash
python app_huecos.py
```

Abrir el navegador en: **http://127.0.0.1:7860**

---

## Flujo de Uso

1. **Cargar imagen** — arrastrar o seleccionar la fotografía de la vía
2. **Configurar parámetros** — altura de cámara y activar/desactivar Claude AI
3. **Analizar** — presionar "Analizar imagen"
4. **Revisar reporte** — severidad, área, intervención y análisis IA por falla
5. **Generar PDF** — descargar reporte oficial con nombre automático por fecha

---

## Resultados por Clase

| Clase | Precisión | Recall | F1 | AP50 | AP50-95 |
|-------|-----------|--------|----|------|---------|
| Hueco | 61.98% | 43.20% | 50.91% | 49.13% | 21.38% |
| Piel de Cocodrilo | 70.08% | 60.52% | 64.95% | 66.09% | 35.99% |
| Grieta | 62.30% | 52.76% | 57.13% | 56.53% | 29.73% |
| Parcheo | 62.37% | 73.66% | 67.54% | 72.73% | 44.85% |

---

## Estándares Aplicados

- **ASTM D6433** — Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys
- **SIGMA UMV** — Sistema de Información para la Gestión del Mantenimiento Vial de Bogotá D.C.

---

## Licencia

Este proyecto fue desarrollado con fines académicos como Trabajo Fin de Estudios del Máster en Inteligencia Artificial de UNIR. Todos los derechos reservados por los autores.

---

*Diagnóstico de la Malla Vial en Bogotá D.C. · UNIR – Máster en Inteligencia Artificial 2026*
