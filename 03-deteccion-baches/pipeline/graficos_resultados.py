#!/usr/bin/env python3
"""
Genera las figuras de la práctica a partir de los resultados del pipeline.

Entradas (dentro de la carpeta de resultados):
  resumen.csv                  -> promedios por métrica (obligatorio)
  metricas_por_imagen.csv      -> métricas por imagen (opcional; habilita las distribuciones)

Salidas (en resultados/graficos/):
  fig1_operadores_borde.png
  fig2_esquinas.png
  fig3_binarizacion.png
  fig4_distribucion_por_imagen.png   (solo si existe metricas_por_imagen.csv)

Uso:
  python pipeline/graficos_resultados.py --resultados resultados
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PURPLE = "#8400E0"
LILAC = "#C58BF2"
GRAY = "#4A4A4A"
OPERATORS = ("sobel", "prewitt", "roberts")
OP_LABEL = {"sobel": "Sobel", "prewitt": "Prewitt", "roberts": "Roberts"}
OP_COLOR = {"sobel": PURPLE, "prewitt": LILAC, "roberts": GRAY}


def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "figure.dpi": 100,
    })


def leer_resumen(path):
    datos = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            datos[row["metrica"]] = float(row["media"])
    return datos


def leer_por_imagen(path):
    with open(path, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    columnas = {}
    for k in filas[0].keys():
        if k == "image":
            continue
        valores = []
        for r in filas:
            try:
                v = float(r[k])
                valores.append(v if not math.isnan(v) else np.nan)
            except (ValueError, TypeError):
                valores.append(np.nan)
        columnas[k] = np.array(valores, dtype=float)
    return columnas


def etiquetar(ax, barras, fmt):
    for b in barras:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points")


def fig_operadores(r, salida):
    fig, ejes = plt.subplots(2, 2, figsize=(9, 7))
    metricas = [
        ("edge_density", "Densidad de bordes", "{:.3f}"),
        ("roi_energy_ratio", "Energía de gradiente en ROI", "{:.3f}"),
        ("roi_edge_precision", "Precisión de bordes en ROI", "{:.3f}"),
        ("time_ms", "Tiempo medio por imagen (ms)", "{:.3f}"),
    ]
    for ax, (sufijo, titulo, fmt) in zip(ejes.flat, metricas):
        valores = [r[f"{op}_{sufijo}"] for op in OPERATORS]
        barras = ax.bar([OP_LABEL[o] for o in OPERATORS], valores,
                        color=[OP_COLOR[o] for o in OPERATORS], width=0.6)
        ax.set_title(titulo)
        etiquetar(ax, barras, fmt)
        # Base en cero: evita exagerar diferencias pequeñas entre operadores
        ax.set_ylim(0, max(valores) * 1.18)
    fig.suptitle("Comparación de operadores de borde (665 imágenes, 512 × 512 px)",
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def fig_esquinas(r, salida):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    detectores = [("shi_tomasi", "Shi-Tomasi"), ("harris", "Harris")]
    nombres = [d[1] for d in detectores]
    puntos = [r[f"{d[0]}_n_points"] for d in detectores]
    barras = ax1.bar(nombres, puntos, color=[PURPLE, LILAC], width=0.55)
    etiquetar(ax1, barras, "{:.1f}")
    ax1.set_title("Puntos de interés por imagen")
    ax1.axhline(150, color=GRAY, linestyle="--", linewidth=1)
    ax1.text(0.5, 165, "límite de\nShi-Tomasi", color=GRAY, fontsize=8,
             ha="center", va="bottom")
    ax1.set_ylabel("Puntos (promedio)")

    x = np.arange(len(detectores))
    ancho = 0.36
    en_roi = [r[f"{d[0]}_in_roi_ratio"] for d in detectores]
    hit = [r[f"{d[0]}_box_hit_rate"] for d in detectores]
    b1 = ax2.bar(x - ancho / 2, en_roi, ancho, color=PURPLE, label="Puntos dentro de la ROI")
    b2 = ax2.bar(x + ancho / 2, hit, ancho, color=LILAC, label="Cajas con al menos un punto")
    etiquetar(ax2, b1, "{:.2f}")
    etiquetar(ax2, b2, "{:.2f}")
    ax2.set_xticks(x, nombres)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Localización de esquinas frente a las anotaciones")
    ax2.legend(frameon=False, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def fig_binarizacion(r, salida):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    metodos = [("otsu", "Otsu global"), ("adaptive", "Umbral adaptativo")]
    nombres = [m[1] for m in metodos]
    iou = [r[f"bin_{m[0]}_iou"] for m in metodos]
    fg = [r[f"bin_{m[0]}_fg_ratio"] for m in metodos]
    b1 = ax1.bar(nombres, iou, color=[PURPLE, LILAC], width=0.55)
    etiquetar(ax1, b1, "{:.3f}")
    ax1.set_title("IoU medio con la máscara de bache")
    ax1.set_ylim(0, max(iou) * 1.35)
    b2 = ax2.bar(nombres, fg, color=[PURPLE, LILAC], width=0.55)
    etiquetar(ax2, b2, "{:.3f}")
    ax2.set_title("Fracción de píxeles en primer plano")
    ax2.set_ylim(0, max(fg) * 1.25)
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def fig_distribuciones(datos, salida):
    fig, ejes = plt.subplots(1, 3, figsize=(12, 4.2))

    iou_otsu = datos["bin_otsu_iou"]
    iou_ada = datos["bin_adaptive_iou"]
    bp = ejes[0].boxplot([iou_otsu[~np.isnan(iou_otsu)], iou_ada[~np.isnan(iou_ada)]],
                         tick_labels=["Otsu", "Adaptativo"], patch_artist=True, widths=0.5)
    for parche, color in zip(bp["boxes"], [PURPLE, LILAC]):
        parche.set_facecolor(color)
        parche.set_alpha(0.7)
    ejes[0].set_title("IoU por imagen (binarización)")

    for ax, sufijo, titulo in [(ejes[1], "roi_energy_ratio", "Energía de gradiente en ROI"),
                               (ejes[2], "roi_edge_precision", "Precisión de bordes en ROI")]:
        serie = [datos[f"{op}_{sufijo}"][~np.isnan(datos[f"{op}_{sufijo}"])] for op in OPERATORS]
        bp = ax.boxplot(serie, tick_labels=[OP_LABEL[o] for o in OPERATORS],
                        patch_artist=True, widths=0.5)
        for parche, op in zip(bp["boxes"], OPERATORS):
            parche.set_facecolor(OP_COLOR[op])
            parche.set_alpha(0.7)
        ax.set_title(titulo)

    fig.suptitle("Distribución de las métricas por imagen", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Gráficos de resultados de la práctica")
    ap.add_argument("--resultados", default="resultados", help="Carpeta con resumen.csv")
    args = ap.parse_args()

    estilo()
    carpeta = os.path.join(args.resultados, "graficos")
    os.makedirs(carpeta, exist_ok=True)

    resumen = os.path.join(args.resultados, "resumen.csv")
    if not os.path.isfile(resumen):
        raise SystemExit(f"No se encontró {resumen}. Ejecuta primero el pipeline.")
    r = leer_resumen(resumen)

    fig_operadores(r, os.path.join(carpeta, "fig1_operadores_borde.png"))
    fig_esquinas(r, os.path.join(carpeta, "fig2_esquinas.png"))
    fig_binarizacion(r, os.path.join(carpeta, "fig3_binarizacion.png"))
    print("Generadas: fig1_operadores_borde.png, fig2_esquinas.png, fig3_binarizacion.png")

    por_imagen = os.path.join(args.resultados, "metricas_por_imagen.csv")
    if os.path.isfile(por_imagen):
        datos = leer_por_imagen(por_imagen)
        fig_distribuciones(datos, os.path.join(carpeta, "fig4_distribucion_por_imagen.png"))
        print("Generada: fig4_distribucion_por_imagen.png")
    else:
        print("Sin metricas_por_imagen.csv: se omite la figura de distribuciones.")

    print(f"Carpeta de salida: {carpeta}")


if __name__ == "__main__":
    main()
