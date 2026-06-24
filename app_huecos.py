import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import anthropic
import base64
import os
import tempfile
from datetime import datetime
from fpdf import FPDF

# ============================================================
# CONFIGURACIÓN
# ============================================================
ANTHROPIC_API_KEY = "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"
MODELO_PATH = r"C:\Scripts\TFE_UMV_Final\Resultados\modelo_final_bogota\weights\best.pt"
LOGO_PATH   = r"C:\Scripts\TFE_UMV_Final\logo-umv-bogota.webp"

modelo      = YOLO(MODELO_PATH)
cliente_llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def _logo_b64():
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

LOGO_B64  = _logo_b64()
LOGO_HTML = (f'<img src="data:image/webp;base64,{LOGO_B64}" style="height:52px;object-fit:contain;" />'
             if LOGO_B64 else "")

CLASES     = {0:"Hueco", 1:"Piel de Cocodrilo", 2:"Grieta", 3:"Parcheo"}
COLORES    = {0:(80,80,255), 1:(80,165,255), 2:(40,120,255), 3:(40,200,120)}

# ============================================================
# REGLAS ASTM D6433
# ============================================================
def sev(cid, a, pct):
    c = CLASES.get(cid,"")
    if c=="Hueco":
        if a is not None:
            if a<=0.25: return "BAJA","Sin Intervención (SI)","🟡"
            if a<=0.64: return "BAJA","Parcheo (PA)","🟡"
            if a<=2.0:  return "MEDIA","Parcheo / Bacheo (BA)","🟠"
            return "ALTA","Bacheo (BA)","🔴"
        return ("BAJA","Parcheo (PA)","🟡") if pct<2 else ("MEDIA","Parcheo / Bacheo","🟠") if pct<8 else ("ALTA","Bacheo (BA)","🔴")
    if c=="Piel de Cocodrilo":
        if a is not None:
            if a<1.0: return "BAJA","Sello de Fisuras (SF)","🟡"
            if a<4.0: return "MEDIA","Parcheo (PA)","🟠"
            return "ALTA","Bacheo (BA)","🔴"
        return ("BAJA","Sello de Fisuras (SF)","🟡") if pct<5 else ("MEDIA","Parcheo (PA)","🟠") if pct<15 else ("ALTA","Bacheo (BA)","🔴")
    if c=="Grieta":
        if a is not None:
            if a<0.5: return "BAJA","Sin Interv. / Sello (SF)","🟡"
            if a<2.0: return "MEDIA","Sello de Fisuras (SF)","🟠"
            return "ALTA","Parcheo (PA)","🔴"
        return ("BAJA","Sin Interv. / Sello (SF)","🟡") if pct<3 else ("MEDIA","Sello de Fisuras (SF)","🟠") if pct<10 else ("ALTA","Parcheo (PA)","🔴")
    if c=="Parcheo":
        if a is not None:
            if a<0.5: return "BAJA","Sin Intervención (SI)","🟡"
            if a<2.0: return "MEDIA","Parcheo (PA)","🟠"
            return "ALTA","Parcheo (PA)","🔴"
        return ("BAJA","Sin Intervención (SI)","🟡") if pct<3 else ("MEDIA","Parcheo (PA)","🟠") if pct<10 else ("ALTA","Parcheo (PA)","🔴")
    return "MEDIA","Revisar en campo","🟠"

def pri(dets):
    if not dets: return "SIN FALLAS","✅"
    s = [d["sev"] for d in dets]
    if "ALTA"  in s: return "INTERVENCIÓN URGENTE","🔴"
    if "MEDIA" in s: return "INTERVENCIÓN PROGRAMADA","🟠"
    return "MONITOREO PREVENTIVO","🟡"

# ============================================================
# LLM
# ============================================================
def llm(rec, clase, conf, sv, interv, a, pct):
    try:
        _, buf = cv2.imencode(".jpg", rec)
        b64 = base64.b64encode(buf).decode()
        area = f"{a:.2f} m2" if a else f"{pct:.1f}% imagen"
        prompt = (f"Experto malla vial Bogota. ASTM D6433/SIGMA UAERMV.\n"
                  f"Falla:{clase}|Conf:{conf:.0%}|Sev:{sv}|Area:{area}|Interv:{interv}\n\n"
                  "Responde SOLO:\nCAUSA: [1 linea]\nSEVERIDAD: [1-5+palabra]\nACCION: [1 linea]\nTIEMPO: [plazo]")
        r = cliente_llm.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=220,
            messages=[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},
                {"type":"text","text":prompt}]}])
        return r.content[0].text.strip()
    except Exception as e:
        return f"[No disponible: {e}]"

# ============================================================
# PDF  2 páginas
# ============================================================
def safe(txt, maxlen=90):
    """Clean text for latin-1 PDF encoding, truncate if too long."""
    t = str(txt).encode('latin-1','replace').decode('latin-1')
    return t[:maxlen] if len(t) > maxlen else t

def pdf(orig, result, dets, fecha):
    fecha_archivo = datetime.now().strftime("%d-%m-%Y")
    nombre_pdf = f"Reporte_Diagnostico_Vial_Automatizado_{fecha_archivo}.pdf"
    tmp_path = os.path.join(tempfile.gettempdir(), nombre_pdf)
    op = tmp_path.replace(".pdf","_o.jpg")
    rp = tmp_path.replace(".pdf","_r.jpg")
    cv2.imwrite(op, cv2.cvtColor(orig,   cv2.COLOR_RGB2BGR))
    cv2.imwrite(rp, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    p = FPDF(); p.set_margins(15,15,15); p.set_auto_page_break(True,15)

    def hdr_band(p, titulo):
        p.set_fill_color(13,33,55); p.rect(0,0,210,30,"F")
        p.set_font("Helvetica","B",13); p.set_text_color(255,255,255)
        p.set_xy(15,7); p.cell(0,7,titulo,ln=True)
        p.set_font("Helvetica","",8); p.set_text_color(168,212,245)
        p.set_xy(15,16); p.cell(0,5,f"UAERMV Bogota D.C.  |  {fecha}  |  ASTM D6433 / SIGMA UMV")

    # PAG 1
    p.add_page(); hdr_band(p,"REPORTE DE DIAGNOSTICO VIAL")
    prioridad, _ = pri(dets); n = len(dets)
    p.set_xy(18,34); p.set_fill_color(236,242,248); p.set_text_color(20,20,20)
    p.set_font("Helvetica","B",10)
    p.cell(85,7,f"  Fallas: {n}",fill=True)
    p.cell(0,7,f"  Prioridad: {prioridad}",fill=True,ln=True); p.ln(3)

    p.set_font("Helvetica","B",8); p.set_text_color(80,80,80)
    p.cell(85,5,"  Imagen original",ln=False); p.cell(0,5,"  Fallas detectadas",ln=True)
    y0=p.get_y(); iw,ih=84,56
    p.image(op,x=18,y=y0,w=iw,h=ih); p.image(rp,x=106,y=y0,w=iw,h=ih)
    p.ln(ih+4); p.set_draw_color(200,210,220); p.line(18,p.get_y(),192,p.get_y()); p.ln(3)

    p.set_font("Helvetica","B",10); p.set_fill_color(220,230,242); p.set_text_color(15,15,15)
    p.cell(0,7,"  DETALLE DE FALLAS",fill=True,ln=True); p.ln(2)
    for i,d in enumerate(dets,1):
        a_txt = f'{d["area"]:.2f} m2' if d["area"] else f'{d["pct"]:.1f}% img'
        p.set_font("Helvetica","B",9); p.set_fill_color(245,248,251)
        p.set_font("Helvetica","B",9); p.set_fill_color(245,248,251)
        line1 = f"  #{i}  {d['clase']}  |  Conf: {d['conf']:.0%}  |  Sev: {d['sev']}"
        line2 = f"       Area: {a_txt}"
        p.cell(0,6,line1[:90],fill=True,ln=True)
        p.set_font("Helvetica","",9); p.cell(0,5,line2,ln=True)
        p.set_font("Helvetica","",9)
        p.cell(0,5,f"       Intervencion: {safe(d['interv'], 80)}",ln=True)
        if d.get("ia"):
            for l in d["ia"].split("\n"):
                l=l.strip()
                if l:
                    p.set_font("Helvetica","I",8); p.set_text_color(80,80,80)
                    p.cell(0,4,f"       {safe(l, 85)}",ln=True)
                    p.set_text_color(15,15,15)
        p.ln(2)

    # PAG 2
    p.add_page(); hdr_band(p,"RESUMEN EJECUTIVO Y CONCLUSIONES")
    p.set_xy(18,34); p.set_text_color(15,15,15)

    conteo={}
    for d in dets: conteo[d["clase"]]=conteo.get(d["clase"],0)+1
    p.set_font("Helvetica","B",10); p.set_fill_color(220,230,242)
    p.cell(0,7,"  RESUMEN POR TIPO",fill=True,ln=True); p.ln(2)
    p.set_font("Helvetica","",9)
    for f,c in conteo.items():
        p.cell(0,5,f"  - {safe(f)}: {c} deteccion{'es' if c>1 else ''}",ln=True)
    p.ln(4)

    p.set_font("Helvetica","B",10); p.set_fill_color(220,230,242)
    p.cell(0,7,"  ACCIONES REQUERIDAS (SIGMA UMV)",fill=True,ln=True); p.ln(2)
    p.set_font("Helvetica","",9)
    for inv in list(set(d["interv"] for d in dets)):
        p.cell(0,5,f"  -> {safe(inv, 80)}",ln=True)
    p.ln(4)

    alta=sum(1 for d in dets if d["sev"]=="ALTA")
    media=sum(1 for d in dets if d["sev"]=="MEDIA")
    baja=sum(1 for d in dets if d["sev"]=="BAJA")
    p.set_font("Helvetica","B",10); p.set_fill_color(220,230,242)
    p.cell(0,7,"  CONCLUSIONES",fill=True,ln=True); p.ln(2)
    p.set_font("Helvetica","",9)
    cls=[
        f"El analisis identifico {n} falla(s) mediante el modelo YOLOv8.",
        f"Distribucion: {alta} ALTA / {media} MEDIA / {baja} BAJA.",
        f"Prioridad global del segmento vial: {prioridad}.",
    ]
    if alta>0: cls.append("Fallas ALTA requieren intervencion urgente en menos de 30 dias.")
    if media>0: cls.append("Fallas MEDIA deben programarse en menos de 90 dias.")
    cls+=[
        "Analisis de imagen validado con Claude AI (Anthropic).",
        "Reporte generado automaticamente por el Sistema de Diagnostico Vial UAERMV.",
    ]
    for c in cls:
        c_clean = c.encode('latin-1','replace').decode('latin-1')
        if len(c_clean) > 150: c_clean = c_clean[:147]+'...'
        p.cell(0,5,f"  - {c_clean[:95]}",ln=True)
    p.ln(6)
    p.set_font("Helvetica","B",9); p.set_fill_color(245,248,251)
    p.cell(0,6,"  LEYENDA",fill=True,ln=True)
    p.set_font("Helvetica","",8); p.set_text_color(80,80,80)
    p.cell(0,4,"  ALTA: urgente<30d   MEDIA: programada<90d   BAJA: monitoreo",ln=True)
    p.ln(6)
    p.set_font("Helvetica","I",7); p.set_text_color(150,150,150)
    p.cell(0,4,"Andres Parra | Marelis Genes | Jaqueline Velasquez  -  UNIR Master IA 2026",ln=True,align="C")
    p.cell(0,4,safe(f"Generado: {fecha}  |  YOLOv8n  |  ASTM D6433"),ln=True,align="C")

    p.output(tmp_path)
    for x in [op,rp]:
        try: os.remove(x)
        except: pass
    return tmp_path

# ============================================================
# ESTADO
# ============================================================
estado = {"orig":None,"result":None,"dets":[],"fecha":""}

def detectar(imagen, altura, usar_llm):
    if imagen is None:
        return None, "⚠  Carga una imagen para analizar."
    res = modelo.predict(imagen, conf=0.10)
    r   = res[0]; img=imagen.copy()
    H,W = img.shape[:2]; AT=H*W
    esc=None
    if altura and altura>0:
        ar=2*altura*0.839; esc=(ar*ar*(H/W))/AT
    dets=[]
    for box in r.boxes:
        cid=int(box.cls[0]); cn=CLASES.get(cid,f"C{cid}")
        cf=float(box.conf[0]); col=COLORES.get(cid,(40,200,120))
        x1,y1,x2,y2=map(int,box.xyxy[0])
        apx=(x2-x1)*(y2-y1); pct=apx/AT*100; am=apx*esc if esc else None
        sv_,interv,emoji=sev(cid,am,pct)
        ia=None
        if usar_llm:
            rec=imagen[y1:y2,x1:x2]
            if rec.size>0: ia=llm(rec,cn,cf,sv_,interv,am,pct)
        dets.append({"clase":cn,"conf":cf,"sev":sv_,"interv":interv,"emoji":emoji,"area":am,"pct":pct,"ia":ia})
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
        lbl=f"{cn} {cf:.0%} | {sv_}"; lw=len(lbl)*9
        cv2.rectangle(img,(x1,y1-30),(x1+lw,y1),col,-1)
        cv2.putText(img,lbl,(x1+5,y1-9),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
    fecha=datetime.now().strftime("%d/%m/%Y %H:%M")
    estado.update({"orig":imagen,"result":img,"dets":dets,"fecha":fecha})
    n=len(dets)
    if n==0: return img,"✅  Sin fallas detectadas."
    prioridad,ep=pri(dets)
    rep=f"Fecha   : {fecha}\nFallas  : {n}   |   Prioridad: {ep} {prioridad}\n"+"─"*52+"\n"
    for i,d in enumerate(dets,1):
        a_t=f'{d["area"]:.2f} m²' if d["area"] else f'{d["pct"]:.1f}% img'
        rep+=f"\n[{i}] {d['clase']}\n    Confianza   : {d['conf']:.0%}\n    Severidad   : {d['emoji']} {d['sev']}\n    Área        : {a_t}\n    Intervención: {d['interv']}\n"
        if d["ia"]:
            rep+="\n    Análisis IA:\n"
            for l in d["ia"].split("\n"):
                if l.strip(): rep+=f"    {l}\n"
    rep+="\n"+"─"*52+"\n"+"Intervenciones:\n"
    for inv in list(set(d["interv"] for d in dets)): rep+=f"  → {inv}\n"
    rep+="\n🔴 ALTA <30d  |  🟠 MEDIA <90d  |  🟡 BAJA — monitoreo"
    return img, rep

def limpiar():
    estado.update({"orig":None,"result":None,"dets":[],"fecha":""})
    return None, None, ""

def exportar():
    if not estado["dets"] or estado["orig"] is None: return None
    return pdf(estado["orig"],estado["result"],estado["dets"],estado["fecha"])

# ============================================================
# CSS COMPLETO — FUERZA TEMA OSCURO EN TODO
# ============================================================
CSS = """
/* reset global */
*, *::before, *::after { box-sizing: border-box !important; }

/* fondo raíz */
html, body, .gradio-container, #root, .main, .wrap {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* panels / cards */
.gr-panel, .gr-box, .gr-form, .block, .svelte-1gfkn6j,
.panel, div.gradio-container > div {
    background-color: #0d1117 !important;
}

/* image components — forzar fondo oscuro */
.gr-image, .image-container, .upload-container,
.svelte-lf36lo, .svelte-1oo6njo,
div[data-testid="image"],
div[data-testid="image"] > div,
div[data-testid="image"] canvas,
.image-preview, .image-frame,
.image-upload-container, .drop-container {
    background-color: #161b22 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 10px !important;
}

/* quitar fondos blancos de cualquier div interno */
div[class*="upload"], div[class*="drop"], div[class*="image"],
div[class*="preview"], div[class*="frame"] {
    background-color: #161b22 !important;
}

/* textbox */
textarea, .gr-text-input, .gr-textbox textarea,
div[data-testid="textbox"] textarea {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
}

/* number input */
input[type=number] {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
}

/* labels */
label, .gr-block-label span, .label-wrap span,
.svelte-1b6s6g, span[class*="label"] {
    color: #7d8590 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.7px !important;
    text-transform: uppercase !important;
}

/* checkbox */
input[type=checkbox] { accent-color: #1f6feb !important; }
.gr-checkbox-label, .checkbox-label { color: #c9d1d9 !important; font-size: 13px !important; }

/* botón primario */
.gr-button-primary, button.primary, button[class*="primary"] {
    background-color: #1f6feb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
}
.gr-button-primary:hover { background-color: #388bfd !important; }

/* botón secundario */
.gr-button-secondary, button.secondary, button[class*="secondary"] {
    background-color: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
.gr-button-secondary:hover { background-color: #30363d !important; }

/* file upload output */
.gr-file, div[data-testid="file"], .file-preview {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* row / column containers */
.gr-row, .gr-column { gap: 16px !important; }

/* scroll bars */
::-webkit-scrollbar { width: 6px; background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* hide gradio footer */
footer, .footer { display: none !important; }

/* fix any remaining white backgrounds */
.block { background-color: #161b22 !important; border: 1px solid #21262d !important; border-radius: 10px !important; }

/* hide file component box, keep download functional */
#pdf-download { display: none !important; }

/* remove dark center overlay on image upload */
.upload-button, .upload-btn, div[class*="upload-button"],
.svelte-1oo6njo > div > div,
div[data-testid="image"] > div > div > div[role="button"],
div[data-testid="image"] .wrap > div:not(.image-container),
.image-upload div.wrap > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* style the drop zone text nicely */
div[data-testid="image"] .wrap {
    background: #161b22 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 10px !important;
}
div[data-testid="image"] .wrap * {
    color: #7d8590 !important;
}
"""

# ============================================================
# INTERFAZ
# ============================================================
with gr.Blocks(css=CSS, theme=gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "sans-serif"],
), title="Diagnostico Vial - UAERMV Bogota") as app:

    header_html = (
        "<div style='background:linear-gradient(135deg,#0d2137 0%,#1a3a5c 100%);"
        "border-radius:12px;padding:20px 28px;margin-bottom:20px;"
        "display:flex;align-items:center;justify-content:space-between;"
        "border:1px solid #1f3a5c;'>"
        "<div style='display:flex;align-items:center;gap:20px;'>"
        "<div style='background:white;border-radius:10px;padding:8px 14px;"
        "display:flex;align-items:center;'>"
        + LOGO_HTML +
        "</div>"
        "<div>"
        "<p style='color:white;margin:0;font-size:20px;font-weight:700;'>"
        "Sistema de Diagnostico Vial Automatizado</p>"
        "<p style='color:#a8d4f5;margin:5px 0 0;font-size:13px;'>"
        "Unidad Administrativa Especial de Rehabilitacion y Mantenimiento Vial - Bogota D.C.</p>"
        "<p style='color:#5a8fa8;margin:3px 0 0;font-size:11px;'>"
        "YOLOv8 + Claude AI &nbsp;|&nbsp; ASTM D6433 / SIGMA UMV</p>"
        "</div></div>"
        "<span style='background:rgba(31,111,235,0.25);color:#79c0ff;"
        "border:1px solid #1f6feb;padding:7px 16px;"
        "border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.8px;'>"
        "TFE - MASTER IA - UNIR 2026</span>"
        "</div>"
    )
    gr.HTML(header_html)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<p style='color:#7d8590;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px;'>01 - Imagen de la via</p>")
            img_in = gr.Image(label="", height=320, sources=["upload","clipboard"])
            with gr.Row():
                altura   = gr.Number(label="Altura camara (m)", value=1.5, precision=1)
                usar_llm = gr.Checkbox(label="Analisis Claude AI", value=True)
            with gr.Row():
                btn_a = gr.Button("Analizar imagen", variant="primary",  size="lg")
                btn_l = gr.Button("Limpiar",          variant="secondary", size="lg")

        with gr.Column(scale=1):
            gr.HTML("<p style='color:#7d8590;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px;'>02 - Fallas detectadas</p>")
            img_out = gr.Image(label="", height=320)



    gr.HTML("<p style='color:#7d8590;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px;'>03 - Reporte de diagnostico</p>")
    reporte = gr.Textbox(label="", lines=16, max_lines=28,
                         placeholder="Carga una imagen y presiona Analizar para generar el reporte...")

    with gr.Row():
        btn_pdf = gr.Button("Generar reporte PDF", variant="secondary", size="lg")
    pdf_out = gr.File(label="", visible=True, elem_id="pdf-download")

    gr.HTML(
        "<div style='text-align:center;padding:16px;color:#484f58;font-size:10px;"
        "border-top:1px solid #21262d;margin-top:20px;'>"
        "Andres Parra &nbsp; Marelis Genes &nbsp; Jaqueline Velasquez"
        " | Diagnostico de la Malla Vial en Bogota D.C."
        " | UNIR - Master en Inteligencia Artificial 2026"
        "</div>"
    )

    btn_a.click(fn=detectar,  inputs=[img_in,altura,usar_llm], outputs=[img_out,reporte])
    btn_l.click(fn=limpiar,   inputs=[],                       outputs=[img_in,img_out,reporte])
    btn_pdf.click(fn=exportar, inputs=[], outputs=[pdf_out])

    pdf_out.change(
        fn=None,
        inputs=[pdf_out],
        outputs=[],
        js="(f) => { if(f && f.url) { const a=document.createElement('a'); a.href=f.url; a.download=f.orig_name||'Reporte_Diagnostico_Vial.pdf'; document.body.appendChild(a); a.click(); document.body.removeChild(a); } }"
    )

app.launch()
