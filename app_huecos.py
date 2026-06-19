import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import anthropic
import base64

# ============================================================
# CONFIGURACIÓN
# ============================================================
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "") 

modelo = YOLO(r"C:\Scripts\UMV\resultados\modelo_final_bogota-2\weights\best.pt")

cliente_llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ============================================================
# CLASES Y COLORES
# ============================================================
CLASES = {
    0: 'Hueco',
    1: 'Piel de Cocodrilo',
    2: 'Grieta',
    3: 'Parcheo'
}

COLORES = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (255, 100, 0),
    3: (0, 200, 100),
}

# ============================================================
# REGLAS DE NEGOCIO ASTM D6433 / SIGMA UMV
# ============================================================
def calcular_severidad_intervencion(clase_id, area_m2, pct_area):
    clase = CLASES.get(clase_id, '')

    if clase == 'Hueco':
        if area_m2 is not None:
            if area_m2 <= 0.25:
                return 'BAJA', 'Sin Intervención (SI)', '🟡'
            elif area_m2 <= 0.64:
                return 'BAJA', 'Parcheo (PA)', '🟡'
            elif area_m2 <= 2.0:
                return 'MEDIA', 'Parcheo (PA) / Bacheo (BA)', '🟠'
            else:
                return 'ALTA', 'Bacheo (BA)', '🔴'
        else:
            if pct_area < 2:
                return 'BAJA', 'Parcheo (PA)', '🟡'
            elif pct_area < 8:
                return 'MEDIA', 'Parcheo (PA) / Bacheo (BA)', '🟠'
            else:
                return 'ALTA', 'Bacheo (BA)', '🔴'

    elif clase == 'Piel de Cocodrilo':
        if area_m2 is not None:
            if area_m2 < 1.0:
                return 'BAJA', 'Sello de Fisuras (SF)', '🟡'
            elif area_m2 < 4.0:
                return 'MEDIA', 'Parcheo (PA)', '🟠'
            else:
                return 'ALTA', 'Bacheo (BA)', '🔴'
        else:
            if pct_area < 5:
                return 'BAJA', 'Sello de Fisuras (SF)', '🟡'
            elif pct_area < 15:
                return 'MEDIA', 'Parcheo (PA)', '🟠'
            else:
                return 'ALTA', 'Bacheo (BA)', '🔴'

    elif clase == 'Grieta':
        if area_m2 is not None:
            if area_m2 < 0.5:
                return 'BAJA', 'Sin Intervención (SI) / Sello de Fisuras (SF)', '🟡'
            elif area_m2 < 2.0:
                return 'MEDIA', 'Sello de Fisuras (SF)', '🟠'
            else:
                return 'ALTA', 'Parcheo (PA)', '🔴'
        else:
            if pct_area < 3:
                return 'BAJA', 'Sin Intervención (SI) / Sello de Fisuras (SF)', '🟡'
            elif pct_area < 10:
                return 'MEDIA', 'Sello de Fisuras (SF)', '🟠'
            else:
                return 'ALTA', 'Parcheo (PA)', '🔴'

    elif clase == 'Parcheo':
        if area_m2 is not None:
            if area_m2 < 0.5:
                return 'BAJA', 'Sin Intervención (SI)', '🟡'
            elif area_m2 < 2.0:
                return 'MEDIA', 'Parcheo (PA)', '🟠'
            else:
                return 'ALTA', 'Parcheo (PA)', '🔴'
        else:
            if pct_area < 3:
                return 'BAJA', 'Sin Intervención (SI)', '🟡'
            elif pct_area < 10:
                return 'MEDIA', 'Parcheo (PA)', '🟠'
            else:
                return 'ALTA', 'Parcheo (PA)', '🔴'

    return 'MEDIA', 'Revisar en campo', '🟠'


def calcular_prioridad_global(detecciones):
    if not detecciones:
        return 'SIN FALLAS', '✅'
    severidades = [d['severidad'] for d in detecciones]
    if 'ALTA' in severidades:
        return 'INTERVENCIÓN URGENTE', '🔴'
    elif 'MEDIA' in severidades:
        return 'INTERVENCIÓN PROGRAMADA', '🟠'
    else:
        return 'MONITOREO', '🟡'


# ============================================================
# ANÁLISIS LLM CON CLAUDE
# ============================================================
def analizar_con_llm(imagen_recortada, clase_nombre, confianza, severidad, intervencion, area_m2, pct_area):
    try:
        # Convertir recorte a base64
        _, buffer = cv2.imencode('.jpg', imagen_recortada)
        imagen_b64 = base64.b64encode(buffer).decode('utf-8')

        area_texto = f"{area_m2:.2f} m²" if area_m2 else f"{pct_area:.1f}% de la imagen"

        prompt = f"""Eres un experto en mantenimiento de malla vial urbana de Bogotá D.C., con conocimiento en metodología ASTM D6433 y sistema SIGMA de la UMV.

YOLO detectó:
- Falla: {clase_nombre}
- Confianza: {confianza:.0%}
- Severidad ASTM: {severidad}
- Área: {area_texto}
- Intervención ASTM: {intervencion}

Analiza la imagen y responde EXACTAMENTE en este formato, sin texto adicional:

CAUSA RAÍZ: [máximo 2 líneas explicando por qué ocurre esta falla]
SEVERIDAD (1-5): [solo el número y una palabra: 1=Mínima, 2=Leve, 3=Moderada, 4=Grave, 5=Crítica]
ACCIÓN RECOMENDADA: [intervención específica según SIGMA UMV en máximo 1 línea]
TIEMPO ESTIMADO: [cuánto tiempo antes de que empeore significativamente]"""

        respuesta = cliente_llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": imagen_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return respuesta.content[0].text

    except Exception as e:
        return f"[Análisis LLM no disponible: {str(e)}]"


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def detectar_fallas(imagen, altura_camara, usar_llm):
    if imagen is None:
        return None, "Por favor carga una imagen"

    resultados = modelo.predict(imagen, conf=0.20)
    r = resultados[0]

    img = imagen.copy()
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    escala_m2 = None
    if altura_camara and altura_camara > 0:
        ancho_real = 2 * altura_camara * 0.839
        alto_real = ancho_real * (img_h / img_w)
        m2_por_pixel = (ancho_real * alto_real) / img_area
        escala_m2 = m2_por_pixel

    detecciones = []

    for box in r.boxes:
        clase_id = int(box.cls[0])
        clase_nombre = CLASES.get(clase_id, f'Clase {clase_id}')
        confianza = float(box.conf[0])
        color = COLORES.get(clase_id, (0, 200, 100))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ancho = x2 - x1
        alto = y2 - y1
        area_px = ancho * alto
        pct_area = (area_px / img_area) * 100

        area_m2 = None
        if escala_m2:
            area_m2 = area_px * escala_m2

        severidad, intervencion, emoji = calcular_severidad_intervencion(
            clase_id, area_m2, pct_area
        )

        # Análisis LLM si está activado
        analisis_llm = None
        if usar_llm:
            recorte = imagen[y1:y2, x1:x2]
            if recorte.size > 0:
                analisis_llm = analizar_con_llm(
                    recorte, clase_nombre, confianza,
                    severidad, intervencion, area_m2, pct_area
                )

        detecciones.append({
            'clase': clase_nombre,
            'confianza': confianza,
            'severidad': severidad,
            'intervencion': intervencion,
            'emoji': emoji,
            'area_m2': area_m2,
            'pct_area': pct_area,
            'analisis_llm': analisis_llm,
        })

        # Dibujar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{clase_nombre} {confianza:.0%} | {severidad}"
        label_w = len(label) * 9
        cv2.rectangle(img, (x1, y1-28), (x1+label_w, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # ============================================================
    # REPORTE
    # ============================================================
    n_fallas = len(detecciones)

    if n_fallas == 0:
        reporte = "✅ No se detectaron fallas en la imagen."
    else:
        prioridad, emoji_prior = calcular_prioridad_global(detecciones)

        reporte = f"{'═'*58}\n"
        reporte += f"  REPORTE DIAGNÓSTICO VIAL — UMV BOGOTÁ\n"
        reporte += f"{'═'*58}\n"
        reporte += f"  Fallas detectadas : {n_fallas}\n"
        reporte += f"  Prioridad global  : {emoji_prior} {prioridad}\n"
        reporte += f"{'═'*58}\n\n"

        reporte += f"DETALLE DE FALLAS:\n"
        reporte += f"{'─'*58}\n"

        for i, d in enumerate(detecciones, 1):
            reporte += f"\n  [{i}] {d['clase']}\n"
            reporte += f"      Confianza    : {d['confianza']:.0%}\n"
            reporte += f"      Severidad    : {d['emoji']} {d['severidad']}\n"
            if d['area_m2'] is not None:
                reporte += f"      Área         : {d['area_m2']:.2f} m²\n"
            else:
                reporte += f"      Área         : {d['pct_area']:.1f}% imagen\n"
            reporte += f"      Intervención : {d['intervencion']}\n"

            if d['analisis_llm']:
                reporte += f"\n      ── ANÁLISIS PROFUNDO (Claude AI) ──\n"
                for linea in d['analisis_llm'].split('\n'):
                    reporte += f"      {linea}\n"

        reporte += f"\n{'─'*58}\n"
        reporte += f"RESUMEN POR TIPO:\n"
        conteo = {}
        for d in detecciones:
            conteo[d['clase']] = conteo.get(d['clase'], 0) + 1
        for falla, cantidad in conteo.items():
            reporte += f"  - {falla}: {cantidad}\n"

        reporte += f"\n{'─'*58}\n"
        reporte += f"INTERVENCIONES REQUERIDAS:\n"
        intervenciones = list(set([d['intervencion'] for d in detecciones]))
        for inv in intervenciones:
            reporte += f"  → {inv}\n"

        reporte += f"\n{'─'*58}\n"
        reporte += f"LEYENDA:\n"
        reporte += f"  🔴 ALTA  — Intervención urgente\n"
        reporte += f"  🟠 MEDIA — Intervención programada\n"
        reporte += f"  🟡 BAJA  — Monitoreo\n"

        if not escala_m2:
            reporte += f"\n⚠️  Ingresa altura de cámara para áreas en m².\n"

    return img, reporte


# ============================================================
# INTERFAZ GRADIO
# ============================================================
interfaz = gr.Interface(
    fn=detectar_fallas,
    inputs=[
        gr.Image(label="📷 Imagen de la vía"),
        gr.Number(label="Altura de cámara en metros (opcional)", value=1.5),
        gr.Checkbox(label="🤖 Activar análisis profundo con Claude AI", value=True)
    ],
    outputs=[
        gr.Image(label="Resultado con fallas detectadas"),
        gr.Textbox(label="Reporte de diagnóstico UMV", lines=35)
    ],
    title="🛣️ Detector de Fallas Viales — UMV Bogotá",
    description="Sistema YOLOv8 + Claude AI para detección automática de fallas viales. Clasifica Hueco, Piel de Cocodrilo, Grieta y Parcheo con severidad e intervención según ASTM D6433 / SIGMA UMV."
)

interfaz.launch()