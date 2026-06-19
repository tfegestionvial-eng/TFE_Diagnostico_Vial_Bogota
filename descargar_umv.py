import requests
import os
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# EDITA SOLO ESTA SECCIÓN
# ============================================================
ids = [308778, 313558, 313563, 296503]
destino = r"C:\Descargas\fotos_umv"
PAUSA_ENTRE_IDS = 2
# ============================================================

API_URL  = "https://sigma.umv.gov.co/SIGMA-backend/api/consulta/fotosMantenimientoVial"
BASE_IMG = "https://sigma.umv.gov.co/Mantenimientos/"

HEADERS_API = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Referer": "https://sigma.umv.gov.co/sigma2/",
}

os.makedirs(destino, exist_ok=True)
resumen = []

for id_mv in ids:
    print(f"\n{'='*50}")
    print(f"Procesando ID: {id_mv}")

    carpeta_id = os.path.join(destino, str(id_mv))
    os.makedirs(carpeta_id, exist_ok=True)

    try:
        # Llamar a la API para obtener la lista de fotos
        payload = {"filtro": f"id_mantenimiento_vial ={id_mv}"}
        resp = requests.post(API_URL, json=payload, headers=HEADERS_API, verify=False, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        fotos = [f for f in data.get("respuesta", []) if f.get("tipo_archivo") == "Fotografía"]
        print(f"Fotografías encontradas: {len(fotos)}")

        descargadas = 0
        for i, foto in enumerate(fotos):
            url_archivo = foto.get("url_archivo", "")
            nombre_archivo = foto.get("nombre_archivo", f"{id_mv}_foto_{i+1}.jpg")
            url_img = BASE_IMG + url_archivo

            try:
                r_img = requests.get(url_img, headers=HEADERS_API, verify=False, timeout=30)
                r_img.raise_for_status()

                # Respetar el nombre original del archivo
                ruta = os.path.join(carpeta_id, nombre_archivo)
                with open(ruta, 'wb') as f:
                    f.write(r_img.content)
                print(f"  ✓ {nombre_archivo}")
                descargadas += 1

            except Exception as e:
                print(f"  ✗ Error descargando {nombre_archivo}: {e}")

        resumen.append((id_mv, descargadas, len(fotos)))

    except Exception as e:
        print(f"  ✗ Error al consultar ID {id_mv}: {e}")
        resumen.append((id_mv, 0, 0))

    time.sleep(PAUSA_ENTRE_IDS)

print(f"\n{'='*50}")
print("RESUMEN FINAL")
print(f"{'='*50}")
for id_mv, desc, total in resumen:
    print(f"  ID {id_mv}: {desc}/{total} fotos descargadas")
print(f"\nArchivos guardados en: {destino}")