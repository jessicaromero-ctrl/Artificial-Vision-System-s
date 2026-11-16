"""Segmentación de hojas de tomate (PVC1, PVC2) con un pipeline clásico de PDI.

Pipeline por imagen:
  1. Lectura en BGR y conversión a HSV.
  2. Suavizado gaussiano para reducir ruido de la cámara.
  3. Umbral de Otsu sobre el croma (Lab): la hoja se aleja del gris neutro del fondo.
  4. Apertura morfológica (elimina motas) y cierre morfológico (une huecos pequeños).
  5. Relleno de huecos interiores y conservación del componente conectado más grande (la hoja).
  6. Máscara binaria final: blanco = hoja, negro = fondo.
  7. Segmentación exploratoria de síntomas (amarillamiento y necrosis) dentro de la hoja.
  8. Métricas: área, porcentaje de la imagen, componentes y, si se dan, IoU y Dice contra una máscara de referencia.
  9. Análisis de sensibilidad: área de la hoja al variar el umbral alrededor del valor de Otsu.

Uso:
    python segmentar_hojas.py --imagenes PVC1.JPG PVC2.JPG --salida resultados_segmentacion
    python segmentar_hojas.py --imagenes PVC1.JPG PVC2.JPG --gt ref1.png ref2.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def leer_imagen(ruta):
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return img


def rellenar_huecos(mascara):
    """Rellena regiones de fondo cerradas dentro de la hoja.

    Se inunda el fondo desde un píxel del borde; lo que no se inunda y no es hoja es un hueco.
    """
    h, w = mascara.shape
    inundada = mascara.copy()
    mascara_ff = np.zeros((h + 2, w + 2), np.uint8)
    borde = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)] \
        + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)]
    for r, c in borde:
        if mascara[r, c] == 0:
            cv2.floodFill(inundada, mascara_ff, (c, r), 255)
            break
    huecos = cv2.bitwise_not(inundada)
    return mascara | huecos


def componente_mayor(mascara):
    """Devuelve solo el componente conectado de mayor área (la hoja)."""
    n, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return mascara, 0
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return np.where(etiquetas == k, 255, 0).astype(np.uint8), n - 1


def canal_croma(img_bgr):
    """Croma en Lab: distancia al gris neutro. Fondo gris ~0; hoja verde o amarilla, alto.

    Es más estable que la saturación HSV frente a sombras y zonas oscuras de la hoja.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    croma = np.sqrt(a * a + b * b)
    return cv2.normalize(croma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def segmentar_hoja(img_bgr, apertura=7, cierre=15, umbral_desplazamiento=0):
    """Pipeline de segmentación de la hoja. Devuelve la máscara y las etapas intermedias."""
    suave = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    croma = canal_croma(suave)

    umbral_otsu, binaria = cv2.threshold(croma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if umbral_desplazamiento:
        umbral = int(np.clip(umbral_otsu + umbral_desplazamiento, 0, 255))
        _, binaria = cv2.threshold(croma, umbral, 255, cv2.THRESH_BINARY)

    k_ap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (apertura, apertura))
    k_ci = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cierre, cierre))
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_ap)
    limpia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, k_ci)
    rellena = rellenar_huecos(limpia)
    mascara, n_componentes = componente_mayor(rellena)

    etapas = {
        "1_original": img_bgr,
        "2_croma": croma,
        "3_otsu": binaria,
        "4_morfologia": limpia,
        "5_mascara_final": mascara,
    }
    return mascara, etapas, float(umbral_otsu), n_componentes


def segmentar_sintomas(img_bgr, mascara_hoja):
    """Segmentación exploratoria de síntomas dentro de la hoja.

    Amarillo: matiz entre 15 y 40 en escala OpenCV (0 a 179) con saturación alta.
    Necrosis: zonas muy oscuras (valor bajo) dentro de la hoja.
    Los umbrales son orientativos y deben ajustarse con la inspección visual.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    amarillo = (h >= 15) & (h <= 40) & (s > 60)
    necrosis = v < 60
    sintoma = ((amarillo | necrosis) & (mascara_hoja > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(sintoma, cv2.MORPH_OPEN, k)


def sensibilidad(img_bgr, deltas=(-30, -20, -10, 0, 10, 20, 30)):
    """Porcentaje de la imagen clasificado como hoja al desplazar el umbral de Otsu."""
    resultados = {}
    total = img_bgr.shape[0] * img_bgr.shape[1]
    for d in deltas:
        mascara, _, _, _ = segmentar_hoja(img_bgr, umbral_desplazamiento=d)
        resultados[f"delta_{d:+d}"] = round(100.0 * np.count_nonzero(mascara) / total, 3)
    return resultados


def iou_dice(prediccion, referencia):
    p = prediccion > 0
    r = referencia > 0
    interseccion = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    iou = interseccion / union if union else 0.0
    dice = 2 * interseccion / (p.sum() + r.sum()) if (p.sum() + r.sum()) else 0.0
    return float(iou), float(dice)


def superposicion(img_bgr, mascara):
    salida = img_bgr.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(salida, contornos, -1, (0, 255, 0), 3)
    return salida


def montaje(etapas, ancho=320):
    paneles = []
    for nombre, im in etapas.items():
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        alto = int(im.shape[0] * ancho / im.shape[1])
        pan = cv2.resize(im, (ancho, alto), interpolation=cv2.INTER_AREA)
        cv2.putText(pan, nombre, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        paneles.append(pan)
    alto_max = max(p.shape[0] for p in paneles)
    paneles = [cv2.copyMakeBorder(p, 0, alto_max - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
               for p in paneles]
    return cv2.hconcat(paneles)


def procesar(ruta, salida, referencia=None):
    img = leer_imagen(ruta)
    nombre = Path(ruta).stem
    mascara, etapas, umbral, n_comp = segmentar_hoja(img)
    sintoma = segmentar_sintomas(img, mascara)

    total = img.shape[0] * img.shape[1]
    area_hoja = int(np.count_nonzero(mascara))
    area_sintoma = int(np.count_nonzero(sintoma))
    metricas = {
        "imagen": Path(ruta).name,
        "dimensiones": [int(img.shape[1]), int(img.shape[0])],
        "umbral_otsu_saturacion": umbral,
        "componentes_antes_de_seleccion": n_comp,
        "area_hoja_px": area_hoja,
        "porcentaje_hoja": round(100.0 * area_hoja / total, 3),
        "area_sintomas_px": area_sintoma,
        "porcentaje_sintomas_sobre_hoja": round(100.0 * area_sintoma / max(area_hoja, 1), 3),
        "sensibilidad_area_hoja_pct": sensibilidad(img),
    }

    if referencia is not None:
        ref = cv2.imread(str(referencia), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"No se pudo leer la referencia: {referencia}")
        if ref.shape != mascara.shape:
            ref = cv2.resize(ref, (mascara.shape[1], mascara.shape[0]), interpolation=cv2.INTER_NEAREST)
        iou, dice = iou_dice(mascara, ref)
        metricas["iou"] = round(iou, 4)
        metricas["dice"] = round(dice, 4)

    salida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida / f"{nombre}_mascara_hoja.png"), mascara)
    cv2.imwrite(str(salida / f"{nombre}_mascara_sintomas.png"), sintoma)
    cv2.imwrite(str(salida / f"{nombre}_superposicion.png"), superposicion(img, mascara))
    etapas_con_sintomas = dict(etapas)
    etapas_con_sintomas["6_sintomas"] = sintoma
    cv2.imwrite(str(salida / f"{nombre}_etapas.png"), montaje(etapas_con_sintomas))
    (salida / f"{nombre}_metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    return metricas


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentación de hojas con PDI clásico.")
    parser.add_argument("--imagenes", type=Path, nargs="+", required=True)
    parser.add_argument("--gt", type=Path, nargs="*", default=None,
                        help="Máscaras de referencia opcionales, en el mismo orden que --imagenes.")
    parser.add_argument("--salida", type=Path, default=Path("resultados_segmentacion"))
    return parser.parse_args()


def main():
    args = parse_args()
    referencias = args.gt if args.gt else [None] * len(args.imagenes)
    if len(referencias) != len(args.imagenes):
        raise SystemExit("--gt debe tener la misma cantidad de archivos que --imagenes.")
    for ruta, ref in zip(args.imagenes, referencias):
        m = procesar(ruta, args.salida, ref)
        print(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"\nResultados en: {args.salida}")


if __name__ == "__main__":
    main()
