import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# MAPEO RDD2022 → 4 clases finales
# 0=D00, 1=D01, 2=D10, 3=D11, 4=D20, 5=D40, 6=D43, 7=D44, 8=D50
# ============================================================
RDD_MAP = {
    0: 2,   # D00 grieta longitudinal → grieta
    1: 2,   # D01 grieta transversal → grieta
    2: 2,   # D10 grieta rejilla → grieta
    3: 2,   # D11 otras grietas → grieta
    4: 1,   # D20 piel cocodrilo → piel_cocodrilo
    5: 0,   # D40 hueco → hueco
    6: 3,   # D43 parcheo línea → parcheo
    7: 3,   # D44 parcheo área → parcheo
    8: -1,  # D50 descartada
}

# ============================================================
# MAPEO BOGOTÁ → 4 clases finales
# 0=depresion, 1=desgaste, 2=desprendimiento, 3=grieta, 4=hueco, 5=parcheo, 6=piel_cocodrilo
# ============================================================
BOG_MAP = {
    0: -1,  # depresion → descartada
    1: -1,  # desgaste → descartada
    2: -1,  # desprendimiento → descartada
    3: 2,   # grieta → grieta
    4: 0,   # hueco → hueco
    5: 3,   # parcheo → parcheo
    6: 1,   # piel_cocodrilo → piel_cocodrilo
}

# ============================================================
# RUTAS
# ============================================================
RDD_BASE = Path(r"C:\Scripts\RDD2022.v1i.yolov8")
BOG_BASE = Path(r"C:\Scripts\Etiquetado_Malla_Vial_Bogota.v3i.yolov8")
OUT_BASE = Path(r"C:\Scripts\dataset_combinado")

def procesar_label(src_file, dst_file, class_map):
    lines_out = []
    with open(src_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            new_cls = class_map.get(cls, -1)
            if new_cls == -1:
                continue
            parts[0] = str(new_cls)
            lines_out.append(' '.join(parts))
    if lines_out:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_file, 'w') as f:
            f.write('\n'.join(lines_out))
        return True
    return False

def copiar_dataset(src_base, dst_base, class_map, prefijo):
    copiadas = 0
    for split in ['train', 'valid', 'test']:
        img_src = src_base / split / 'images'
        lbl_src = src_base / split / 'labels'
        img_dst = dst_base / split / 'images'
        lbl_dst = dst_base / split / 'labels'

        if not img_src.exists():
            continue

        for img_file in img_src.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            lbl_file = lbl_src / (img_file.stem + '.txt')
            if not lbl_file.exists():
                continue

            nuevo_nombre = f"{prefijo}_{img_file.name}"
            dst_lbl = lbl_dst / (prefijo + '_' + img_file.stem + '.txt')

            ok = procesar_label(lbl_file, dst_lbl, class_map)
            if ok:
                img_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, img_dst / nuevo_nombre)
                copiadas += 1

    print(f"  {prefijo}: {copiadas} imagenes copiadas")
    return copiadas

def crear_yaml(dst_base):
    yaml_content = f"""train: {dst_base / 'train' / 'images'}
val: {dst_base / 'valid' / 'images'}
test: {dst_base / 'test' / 'images'}

nc: 4
names: ['hueco', 'piel_cocodrilo', 'grieta', 'parcheo']
"""
    yaml_path = dst_base / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

if __name__ == '__main__':
    # Limpiar y crear carpeta combinada
    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)
    OUT_BASE.mkdir(parents=True)

    print("=== COMBINANDO DATASETS ===")
    print("Procesando RDD2022...")
    n1 = copiar_dataset(RDD_BASE, OUT_BASE, RDD_MAP, 'rdd')
    print("Procesando Bogota...")
    n2 = copiar_dataset(BOG_BASE, OUT_BASE, BOG_MAP, 'bog')
    print(f"\nTotal imagenes combinadas: {n1 + n2}")

    yaml_path = crear_yaml(OUT_BASE)
    print(f"data.yaml creado en: {yaml_path}")

    print("\n=== INICIANDO ENTRENAMIENTO MODELO FINAL ===")
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=16,
        name='modelo_final_bogota',
        project=r'C:\Scripts\UMV\resultados',
        device=0
    )

    print("\n✅ Entrenamiento completado!")
    print("Modelo final en: C:\\Scripts\\UMV\\resultados\\modelo_final_bogota\\weights\\best.pt")