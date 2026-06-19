from pathlib import Path
from ultralytics import YOLO
import csv

if __name__ == '__main__':

    MODELO = r"C:\Scripts\UMV\resultados\modelo_final_bogota-2\weights\best.pt"
    DATASET = r"C:\Scripts\dataset_combinado\data.yaml"
    RESULTADOS = r"C:\Scripts\UMV\resultados\modelo_final_bogota-2"
    CLASES = ['hueco', 'piel_cocodrilo', 'grieta', 'parcheo']

    print("=== EVALUACIÓN MODELO FINAL UMV BOGOTÁ ===\n")
    model = YOLO(MODELO)

    metrics = model.val(
        data=DATASET,
        split='test',
        plots=True,
        workers=0,
    )

    print("\n=== EXPORTANDO MÉTRICAS A CSV ===")

    filas = []
    for i, clase in enumerate(CLASES):
        fila = {
            'Clase': clase,
            'Precision': round(float(metrics.box.p[i]), 4),
            'Recall': round(float(metrics.box.r[i]), 4),
            'AP50': round(float(metrics.box.ap50[i]), 4),
            'AP50-95': round(float(metrics.box.ap[i]), 4),
            'F1': round(2 * float(metrics.box.p[i]) * float(metrics.box.r[i]) /
                       (float(metrics.box.p[i]) + float(metrics.box.r[i]) + 1e-9), 4)
        }
        filas.append(fila)

    filas.append({
        'Clase': 'TOTAL (mean)',
        'Precision': round(float(metrics.box.mp), 4),
        'Recall': round(float(metrics.box.mr), 4),
        'AP50': round(float(metrics.box.map50), 4),
        'AP50-95': round(float(metrics.box.map), 4),
        'F1': round(float(metrics.box.f1.mean()), 4)
    })

    csv_path = Path(RESULTADOS) / 'metricas_por_clase.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Clase','Precision','Recall','AP50','AP50-95','F1'])
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n✅ CSV guardado en: {csv_path}")
    print(f"\n{'Clase':<20} {'P':>8} {'R':>8} {'AP50':>8} {'AP50-95':>8} {'F1':>8}")
    print("─" * 60)
    for fila in filas:
        print(f"{fila['Clase']:<20} {fila['Precision']:>8} {fila['Recall']:>8} {fila['AP50']:>8} {fila['AP50-95']:>8} {fila['F1']:>8}")