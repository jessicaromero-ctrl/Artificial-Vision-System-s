#!/usr/bin/env python3
"""
Pipeline de PDI para detección de baches (Pothole Detection, Kaggle / andrewmvd).

Etapas:
  1. Cambio de espacio de color: BGR -> escala de grises.
  2. Estandarización de dimensiones: letterbox a SIZE x SIZE (conserva relación de aspecto).
  3. Suavizado: Gaussiano (por defecto), mediana o bilateral.
  4. Detección de bordes con Sobel, Prewitt y Roberts, con comparación cuantitativa.
  5. Detección de esquinas y puntos de interés: Harris y Shi-Tomasi.
  6. Binarización: Otsu global y umbral adaptativo, con limpieza morfológica.

Evaluación (si existen anotaciones VOC XML): las cajas de bache se convierten en una
máscara de referencia para medir la localización de bordes, esquinas y binarización.

Uso:
  python pipeline/pothole_pipeline.py --images RUTA/images --annotations RUTA/annotations --out resultados
"""
import argparse
import csv
import glob
import os
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

OPERATORS = ("sobel", "prewitt", "roberts")
SMOOTH_METHODS = ("gaussian", "median", "bilateral")
PURPLE_BGR = (224, 0, 132)  # #8400e0 en BGR

# Núcleos de convolución (correlación con cv2.filter2D)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = PREWITT_X.T.copy()
ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float32)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Utilidades de anotación y métricas
# ---------------------------------------------------------------------------
def load_boxes(xml_path):
    """Lee cajas (xmin, ymin, xmax, ymax) de un archivo PASCAL VOC."""
    if not os.path.isfile(xml_path):
        return []
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        coords = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append(tuple(coords))
    return boxes


def safe_div(num, den):
    return float(num) / float(den) if den else float("nan")


def nanmean(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return safe_div(inter, union)


def boxes_to_mask(boxes, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[int(max(0, y1)):int(min(shape[0], y2)), int(max(0, x1)):int(min(shape[1], x2))] = 1
    return mask


# ---------------------------------------------------------------------------
# Etapa 1: espacio de color
# ---------------------------------------------------------------------------
def to_grayscale(bgr):
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Etapa 2: estandarización de dimensiones (letterbox)
# ---------------------------------------------------------------------------
def letterbox(gray, boxes, size, pad_value=114):
    h, w = gray.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    new_boxes = [(x1 * scale + px, y1 * scale + py, x2 * scale + px, y2 * scale + py)
                 for x1, y1, x2, y2 in boxes]
    return canvas, new_boxes


# ---------------------------------------------------------------------------
# Etapa 3: suavizado
# ---------------------------------------------------------------------------
def smooth(gray, method="gaussian"):
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    if method == "median":
        return cv2.medianBlur(gray, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    raise ValueError(f"Método de suavizado no soportado: {method}")


# ---------------------------------------------------------------------------
# Etapa 4: bordes (Sobel, Prewitt, Roberts)
# ---------------------------------------------------------------------------
def gradient_magnitude(img, op):
    f = img.astype(np.float32)
    if op == "sobel":
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    elif op == "prewitt":
        gx = cv2.filter2D(f, cv2.CV_32F, PREWITT_X)
        gy = cv2.filter2D(f, cv2.CV_32F, PREWITT_Y)
    elif op == "roberts":
        gx = cv2.filter2D(f, cv2.CV_32F, ROBERTS_X, anchor=(0, 0))
        gy = cv2.filter2D(f, cv2.CV_32F, ROBERTS_Y, anchor=(0, 0))
    else:
        raise ValueError(f"Operador no soportado: {op}")
    return cv2.magnitude(gx, gy)


def binary_edges(mag):
    """Normaliza la magnitud a uint8 y binariza con umbral de Otsu."""
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges


# ---------------------------------------------------------------------------
# Etapa 5: esquinas y puntos de interés
# ---------------------------------------------------------------------------
def detect_corners(gray, max_corners=150):
    harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    local_max = harris == cv2.dilate(harris, np.ones((5, 5), np.uint8))
    thr = 0.01 * harris.max()
    hy, hx = np.nonzero((harris > thr) & local_max) if harris.max() > 0 else (np.array([]), np.array([]))
    harris_pts = list(zip(hx.tolist(), hy.tolist()))

    shi = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    shi_pts = [] if shi is None else [tuple(p) for p in shi.reshape(-1, 2).astype(int).tolist()]
    return {"harris": harris_pts, "shi_tomasi": shi_pts}


def corner_metrics(points, boxes):
    """Proporción de puntos dentro de cajas y proporción de cajas con al menos un punto."""
    if not points:
        return float("nan"), float("nan")
    inside = [any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in boxes) for x, y in points]
    in_ratio = safe_div(sum(inside), len(points)) if boxes else float("nan")
    if not boxes:
        return in_ratio, float("nan")
    hit = [any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in points) for x1, y1, x2, y2 in boxes]
    return in_ratio, safe_div(sum(hit), len(boxes))


# ---------------------------------------------------------------------------
# Etapa 6: binarización
# ---------------------------------------------------------------------------
def binarize(gray, method="otsu"):
    """Devuelve máscara binaria donde 255 = región oscura (candidata a bache)."""
    if method == "otsu":
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 10)
    else:
        raise ValueError(f"Método de binarización no soportado: {method}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    return bin_img


# ---------------------------------------------------------------------------
# Procesamiento de una imagen
# ---------------------------------------------------------------------------
def process_image(path, raw_boxes, size, smooth_method):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None

    gray = to_grayscale(bgr)                                   # Etapa 1
    gray, boxes = letterbox(gray, raw_boxes, size)             # Etapa 2
    mask = boxes_to_mask(boxes, gray.shape)
    gs = smooth(gray, smooth_method)                           # Etapa 3

    record = {"image": os.path.basename(path), "n_boxes": len(boxes)}
    edge_maps = {}

    for op in OPERATORS:                                       # Etapa 4
        t0 = time.perf_counter()
        mag = gradient_magnitude(gs, op)
        edges = binary_edges(mag)
        elapsed = (time.perf_counter() - t0) * 1000.0
        edge_maps[op] = edges
        edge_px = edges > 0
        record[f"{op}_time_ms"] = elapsed
        record[f"{op}_edge_density"] = float(edge_px.mean())
        record[f"{op}_roi_energy_ratio"] = safe_div(mag[mask > 0].sum(), mag.sum()) if mask.any() else float("nan")
        record[f"{op}_roi_edge_precision"] = safe_div(edge_px[mask > 0].sum(), edge_px.sum()) if mask.any() else float("nan")

    corners = detect_corners(gs)                               # Etapa 5
    for name, pts in corners.items():
        in_ratio, box_hit = corner_metrics(pts, boxes)
        record[f"{name}_n_points"] = len(pts)
        record[f"{name}_in_roi_ratio"] = in_ratio
        record[f"{name}_box_hit_rate"] = box_hit

    for method in ("otsu", "adaptive"):                        # Etapa 6
        bin_img = binarize(gs, method)
        record[f"bin_{method}_iou"] = iou(bin_img > 0, mask > 0) if mask.any() else float("nan")
        record[f"bin_{method}_fg_ratio"] = float((bin_img > 0).mean())

    viz = {"gray": gray, "smooth": gs, "corners": corners, "edges": edge_maps,
           "bin": binarize(gs, "adaptive"), "boxes": boxes}
    return record, viz


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def save_panel(name, viz, out_dir, tile=256):
    def to_tile(img, color=False):
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        return img if color else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    base = viz["gray"].copy()
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in viz["boxes"]:
        cv2.rectangle(base_bgr, (int(x1), int(y1)), (int(x2), int(y2)), PURPLE_BGR, 2)

    corner_img = cv2.cvtColor(viz["smooth"], cv2.COLOR_GRAY2BGR)
    for x, y in viz["corners"]["shi_tomasi"]:
        cv2.circle(corner_img, (x, y), 3, PURPLE_BGR, -1)
    for x, y in viz["corners"]["harris"]:
        cv2.circle(corner_img, (x, y), 2, (0, 255, 255), -1)

    row1 = [base_bgr, viz["smooth"], viz["bin"], corner_img]
    row2 = [viz["edges"][op] for op in OPERATORS]
    row2.append(np.zeros_like(viz["gray"]))
    rows = [np.hstack([to_tile(im, color=im.ndim == 3) for im in row1]),
            np.hstack([to_tile(im) for im in row2])]
    cv2.imwrite(os.path.join(out_dir, f"{os.path.splitext(name)[0]}.jpg"), np.vstack(rows))


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def summarize(records):
    keys = [k for k in records[0] if k not in ("image", "n_boxes")]
    return {k: nanmean([r[k] for r in records]) for k in keys}


def print_operator_table(summary):
    print("\nComparación de operadores de borde (medias):")
    print(f"{'Operador':<10}{'t (ms)':>10}{'Densidad':>12}{'Energía ROI':>14}{'Precisión ROI':>16}")
    for op in OPERATORS:
        print(f"{op:<10}"
              f"{summary[op + '_time_ms']:>10.3f}"
              f"{summary[op + '_edge_density']:>12.4f}"
              f"{summary[op + '_roi_energy_ratio']:>14.4f}"
              f"{summary[op + '_roi_edge_precision']:>16.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pipeline PDI para detección de baches")
    ap.add_argument("--images", required=True, help="Carpeta con las imágenes")
    ap.add_argument("--annotations", default=None, help="Carpeta con XML VOC (opcional)")
    ap.add_argument("--out", default="resultados_pothole", help="Carpeta de salida")
    ap.add_argument("--size", type=int, default=512, help="Lado de la imagen estandarizada")
    ap.add_argument("--smooth", default="gaussian", choices=SMOOTH_METHODS)
    ap.add_argument("--limit", type=int, default=0, help="Procesar solo N imágenes (0 = todas)")
    ap.add_argument("--viz", type=int, default=5, help="Número de paneles de visualización")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    viz_dir = os.path.join(args.out, "paneles")
    os.makedirs(viz_dir, exist_ok=True)

    paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                   for p in glob.glob(os.path.join(args.images, ext)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"No se encontraron imágenes en {args.images}")

    records = []
    for i, path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        raw_boxes = load_boxes(os.path.join(args.annotations, stem + ".xml")) if args.annotations else []
        record, viz = process_image(path, raw_boxes, args.size, args.smooth)
        if record is None:
            print(f"[omitida] no legible: {path}")
            continue
        records.append(record)
        if i < args.viz:
            save_panel(os.path.basename(path), viz, viz_dir)

    if not records:
        raise SystemExit("Ninguna imagen fue procesada correctamente.")

    per_image = os.path.join(args.out, "metricas_por_imagen.csv")
    with open(per_image, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = summarize(records)
    summary_path = os.path.join(args.out, "resumen.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metrica", "media"])
        for k, v in summary.items():
            writer.writerow([k, f"{v:.6f}"])

    print(f"Imágenes procesadas: {len(records)} de {len(paths)}")
    print(f"Tamaño estandarizado: {args.size}x{args.size} | suavizado: {args.smooth}")
    print_operator_table(summary)
    print("\nEsquinas (media):")
    print(f"  Shi-Tomasi: {summary['shi_tomasi_n_points']:.1f} puntos | "
          f"en ROI {summary['shi_tomasi_in_roi_ratio']:.3f} | cajas con punto {summary['shi_tomasi_box_hit_rate']:.3f}")
    print(f"  Harris:     {summary['harris_n_points']:.1f} puntos | "
          f"en ROI {summary['harris_in_roi_ratio']:.3f} | cajas con punto {summary['harris_box_hit_rate']:.3f}")
    print("\nBinarización (IoU con máscara de bache):")
    print(f"  Otsu:      {summary['bin_otsu_iou']:.4f}")
    print(f"  Adaptativa:{summary['bin_adaptive_iou']:.4f}")
    print(f"\nArchivos: {per_image} | {summary_path} | {viz_dir}/")


if __name__ == "__main__":
    main()
