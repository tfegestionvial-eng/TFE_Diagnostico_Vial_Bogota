# TFE — Diagnóstico de la Malla Vial en Bogotá D.C.
## Mediante Visión por Computadora y Redes Neuronales Convolucionales

**Universidad Internacional de La Rioja (UNIR) — Máster en Inteligencia Artificial 2026**

**Autores:** Andrés Felipe Parra Forero · Marelis del Carmen Díaz Genes · María Jaqueline Velásquez Parrado

**Directora:** María Inmaculada Mohino Herranz

---

## Descripción

Sistema de detección automática de fallas viales en Bogotá D.C. basado en YOLOv8 + Claude AI. Detecta y clasifica cuatro tipos de fallas: **hueco**, **piel de cocodrilo**, **grieta** y **parcheo**, aplicando reglas de negocio ASTM D6433 / SIGMA UMV para determinar severidad e intervención recomendada.

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `app_huecos.py` | Interfaz Gradio con pipeline YOLOv8 + Claude AI |
| `entrenar_yolo.py` | Combinación de datasets y entrenamiento YOLOv8 |
| `evaluar_modelo.py` | Evaluación del modelo y exportación de métricas CSV |
| `descargar_umv.py` | Descarga de imágenes del repositorio UMV |

## Requisitos

```bash
pip install ultralytics gradio anthropic opencv-python
```

## Ejecución

```bash
# Configurar API key de Anthropic
$env:ANTHROPIC_API_KEY = "tu-api-key"

# Ejecutar la app
python app_huecos.py
```

## Datasets utilizados

- **RDD2022** — Road Damage Dataset (Japón, India, República Checa)
- **Corpus Bogotano** — Imágenes capturadas por Ingenieros Diagnosticadores de la UAERMV

## Modelo

- Arquitectura: YOLOv8n (Ultralytics)
- Dataset combinado: ~26,000 imágenes
- mAP@50 global: 59.8%
- Entrenamiento: 100 épocas, GPU RTX 4070
