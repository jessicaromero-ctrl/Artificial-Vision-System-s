#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Práctica: operaciones aritméticas y lógicas a nivel de píxel
============================================================
SUMA, RESTA, AND y OR sobre un banco de 7 imágenes dermatológicas
(DermNet / Te Whatu Ora Health New Zealand, página "Neurofibromatosis images").

Uso
---
    python practica_operaciones_pixeles.py \
        --entrada /ruta/a/imagenes  --salida resultados  --tam 512

Requisitos: Python >= 3.9, numpy, opencv-python, matplotlib.
Las imágenes deben llamarse img1.png ... img7.png dentro de --entrada.

Salidas (dentro de --salida)
----------------------------
    figuras/      figuras compuestas numeradas (fig01 ... fig16)
    imagenes/     cada imagen resultado por separado (PNG sin pérdida)
    metricas.csv  todas las métricas calculadas (formato largo)

Convenciones
------------
* Todas las imágenes se procesan como uint8, BGR (convención de OpenCV).
* "Saturada"  : el resultado se recorta al rango [0, 255]   (cv2.add / cv2.subtract).
* "Modular"   : el resultado se reduce módulo 256           (comportamiento de un
                registro de 8 bits sin protección de desbordamiento).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # sin ventana: sólo escribe archivos
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Configuración del dataset
# --------------------------------------------------------------------------- #
PURPURA = "#8400e0"

# recorte_izq: píxeles a recortar por la izquierda para eliminar el fragmento de
# marca de agua DermNet que traen img2, img4 e img6 (ocupa ~32 px del borde).
IMAGENES = {
    1: dict(desc="Mácula extensa en extremidad (fondo oscuro)", tipo="Mácula", recorte_izq=0),
    2: dict(desc="Mácula con lesiones satélite, tono violáceo", tipo="Mácula", recorte_izq=40),
    3: dict(desc="Nódulos cutáneos, plano cercano", tipo="Nódulos", recorte_izq=0),
    4: dict(desc="Dos máculas sobre piel clara", tipo="Mácula", recorte_izq=40),
    5: dict(desc="Nódulos cutáneos (misma escena que img3)", tipo="Nódulos", recorte_izq=0),
    6: dict(desc="Mácula única, tono ocre", tipo="Mácula", recorte_izq=40),
    7: dict(desc="Nódulos múltiples, piel con pliegues", tipo="Nódulos", recorte_izq=0),
}

# Pares de trabajo (elegidos por el contenido, no al azar)
PARES = {
    "P1": (3, 5),  # misma escena, distinta escala de recorte -> resta / registro
    "P2": (4, 6),  # dos máculas claras y brillantes        -> saturación vs módulo
    "P3": (1, 7),  # clases distintas, rangos dinámicos muy distintos
}

METRICAS = []  # (operación, contexto, métrica, valor)


def reg(op, ctx, met, val):
    METRICAS.append((op, ctx, met, float(val)))


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def titulo(txt):
    print("\n" + "=" * 78 + f"\n{txt}\n" + "=" * 78)


def cargar_y_preprocesar(ruta, recorte_izq, tam):
    """Lee, descarta canal alfa, recorta marca de agua, recorta a cuadrado
    centrado y reescala a tam x tam con interpolación por área."""
    im = cv2.imread(str(ruta), cv2.IMREAD_COLOR)  # IMREAD_COLOR descarta alfa
    if im is None:
        sys.exit(f"[ERROR] No se pudo leer {ruta}")
    original = im.copy()
    im = im[:, recorte_izq:]
    h, w = im.shape[:2]
    lado = min(h, w)
    y0, x0 = (h - lado) // 2, (w - lado) // 2
    im = im[y0:y0 + lado, x0:x0 + lado]
    return original, cv2.resize(im, (tam, tam), interpolation=cv2.INTER_AREA)


def gris(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def iou(ma, mb):
    """Jaccard entre máscaras binarias: |A AND B| / |A OR B|."""
    inter = cv2.countNonZero(cv2.bitwise_and(ma, mb))
    union = cv2.countNonZero(cv2.bitwise_or(ma, mb))
    return inter / union if union else 1.0


def guardar_img(ruta, im):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), im)


# --------------------------------------------------------------------------- #
# Operaciones (cada una en una función pequeña y documentada)
# --------------------------------------------------------------------------- #
def suma_saturada(a, b):
    return cv2.add(a, b)


def suma_modular(a, b):
    return (a.astype(np.uint16) + b.astype(np.uint16)).astype(np.uint8)  # mod 256


def suma_ponderada(a, b, alpha):
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0)


def suma_escalar_sat(a, k):
    # cv2.add(a, k) con escalar sólo afectaría al primer canal; se usa un arreglo.
    return cv2.add(a, np.full_like(a, k))


def suma_escalar_mod(a, k):
    return (a.astype(np.uint16) + k).astype(np.uint8)


def resta_saturada(a, b):
    return cv2.subtract(a, b)


def resta_modular(a, b):
    return np.mod(a.astype(np.int16) - b.astype(np.int16), 256).astype(np.uint8)


def diferencia_abs(a, b):
    return cv2.absdiff(a, b)


def op_and(a, b):
    return cv2.bitwise_and(a, b)


def op_or(a, b):
    return cv2.bitwise_or(a, b)


def mascara_lesion(img, sigma_fino=3.0, sigma_fondo=35.0, ganancia=2.0):
    """Máscara binaria de lesión.
    1) Canal L* (Lab) invertido: lo oscuro/pigmentado pasa a valores altos.
    2) Corrección de fondo por RESTA: (L suavizado) - (L muy suavizado).
    3) Umbral de Otsu sobre el residuo recentrado en 128.
    4) Apertura y cierre morfológicos (elimina motas, rellena huecos)."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    inv = 255.0 - L.astype(np.float32)
    fino = cv2.GaussianBlur(inv, (0, 0), sigma_fino)
    fondo = cv2.GaussianBlur(inv, (0, 0), sigma_fondo)
    res = np.clip((fino - fondo) * ganancia + 128, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m, res


def registrar_ecc(ref, mov):
    """Alinea `mov` sobre `ref` con un modelo afín (maximiza el coeficiente de
    correlación mejorado, ECC). Devuelve (imagen_registrada, matriz, cc)."""
    g_ref = gris(ref).astype(np.float32)
    g_mov = gris(mov).astype(np.float32)
    W = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    cc, W = cv2.findTransformECC(g_ref, g_mov, W, cv2.MOTION_AFFINE, crit, None, 5)
    h, w = ref.shape[:2]
    reg_ = cv2.warpAffine(mov, W, (w, h),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)
    return reg_, W, cc


# --------------------------------------------------------------------------- #
# Utilidades de figuras
# --------------------------------------------------------------------------- #
def mostrar(ax, im, txt, fs=8):
    if im.ndim == 3:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(txt, fontsize=fs)
    ax.axis("off")


def hist_gris(ax, im, etiqueta, color, alpha=0.55):
    ax.hist(gris(im).ravel(), bins=64, range=(0, 256), log=True,
            color=color, alpha=alpha, label=etiqueta)


def cerrar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   figura -> {ruta.name}")


def ocultar_sobrantes(axs, n_usados):
    for ax in list(axs.ravel())[n_usados:]:
        ax.axis("off")


# --------------------------------------------------------------------------- #
# Verificación numérica de identidades (asserts reales sobre los datos)
# --------------------------------------------------------------------------- #
def verificar_identidades(nombre, a, b):
    a16, b16 = a.astype(np.uint16), b.astype(np.uint16)
    and_, or_ = op_and(a, b).astype(np.uint16), op_or(a, b).astype(np.uint16)

    ok1 = np.array_equal(a16 + b16, or_ + and_)                     # a+b = (a|b)+(a&b)
    ok2 = bool(np.all(and_ <= np.minimum(a16, b16)))                # a&b <= min
    ok3 = bool(np.all(or_ >= np.maximum(a16, b16)))                 # a|b >= max
    sin_desb = (a16 + b16) <= 255
    ok4 = np.array_equal(suma_saturada(a, b)[sin_desb], suma_modular(a, b)[sin_desb])
    ok5 = np.array_equal(diferencia_abs(a, b),
                         cv2.add(resta_saturada(a, b), resta_saturada(b, a)))
    ok6 = (np.array_equal(op_and(a, np.full_like(a, 255)), a)
           and np.array_equal(op_or(a, np.zeros_like(a)), a))       # elementos neutros
    print(f"   {nombre}:  a+b=(a|b)+(a&b) {ok1} | a&b<=min {ok2} | a|b>=max {ok3} | "
          f"sat==mod sin desborde {ok4} | |a-b|=(a-b)+(b-a) {ok5} | neutros {ok6}")
    assert all([ok1, ok2, ok3, ok4, ok5, ok6]), "Falló una identidad"


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default="/mnt/user-data/uploads", type=Path)
    ap.add_argument("--salida", default="resultados", type=Path)
    ap.add_argument("--tam", default=512, type=int, help="lado del cuadrado de trabajo")
    args = ap.parse_args()

    figs = args.salida / "figuras"
    ims = args.salida / "imagenes"
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    titulo("0. CARGA Y PREPROCESAMIENTO")
    orig, prep = {}, {}
    for i, meta in IMAGENES.items():
        orig[i], prep[i] = cargar_y_preprocesar(args.entrada / f"img{i}.png",
                                                meta["recorte_izq"], args.tam)
        h, w = orig[i].shape[:2]
        print(f"   img{i}: original {w}x{h}  ->  {args.tam}x{args.tam}  "
              f"(recorte_izq={meta['recorte_izq']:>2} px)  media BGR="
              f"{np.round(prep[i].reshape(-1, 3).mean(0), 1)}")
        guardar_img(ims / f"prep_img{i}.png", prep[i])
        reg("preproc", f"img{i}", "ancho_original", w)
        reg("preproc", f"img{i}", "alto_original", h)

    # Fig 01 originales
    fig, axs = plt.subplots(2, 4, figsize=(8, 4.3))
    for ax, (i, im) in zip(axs.ravel(), orig.items()):
        mostrar(ax, im, f"img{i}  ({im.shape[1]}×{im.shape[0]})")
    ocultar_sobrantes(axs, 7)
    cerrar(fig, figs / "fig01_dataset_original.png")

    # Fig 02 preprocesadas
    fig, axs = plt.subplots(2, 4, figsize=(8, 4.3))
    for ax, (i, im) in zip(axs.ravel(), prep.items()):
        mostrar(ax, im, f"img{i}  ({args.tam}×{args.tam})")
    ocultar_sobrantes(axs, 7)
    cerrar(fig, figs / "fig02_dataset_preprocesado.png")

    # Fig 03 matriz de MAE entre todas las imágenes
    M = np.zeros((7, 7))
    for i in range(1, 8):
        for j in range(1, 8):
            M[i - 1, j - 1] = mae(prep[i], prep[j])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("p", ["#ffffff", PURPURA])
    im_ = ax.imshow(M, cmap=cmap)
    ax.set_xticks(range(7), [f"img{k}" for k in range(1, 8)], fontsize=8)
    ax.set_yticks(range(7), [f"img{k}" for k in range(1, 8)], fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.max() * 0.55 else "black")
    ax.set_title("Error absoluto medio (MAE) entre pares de imágenes", fontsize=9)
    fig.colorbar(im_, ax=ax, fraction=0.046, label="MAE (niveles, 0–255)")
    cerrar(fig, figs / "fig03_matriz_mae.png")
    print(f"   MAE(img3,img5) = {M[2, 4]:.2f}   |   mínimo fuera de la diagonal en otros pares = "
          f"{np.min(M[np.triu_indices(7, 1)][M[np.triu_indices(7, 1)] > M[2, 4]]):.2f}")
    for i in range(7):
        for j in range(i + 1, 7):
            reg("preproc", f"img{i + 1}-img{j + 1}", "MAE", M[i, j])

    # ------------------------------------------------------------------ #
    titulo("1. VERIFICACIÓN DE IDENTIDADES (sobre los datos reales)")
    for nom, (i, j) in PARES.items():
        verificar_identidades(f"{nom} (img{i},img{j})", prep[i], prep[j])

    # ------------------------------------------------------------------ #
    titulo("2. SUMA")
    # 2a) Par P2: saturada vs modular vs ponderada
    i, j = PARES["P2"]
    A, B = prep[i], prep[j]
    s_sat, s_mod, s_pon = suma_saturada(A, B), suma_modular(A, B), suma_ponderada(A, B, 0.5)
    frac_sat = np.mean((A.astype(np.uint16) + B.astype(np.uint16)) > 255)
    print(f"   P2 (img{i}+img{j}): valores que desbordan 255 = {100 * frac_sat:.2f} %")
    print(f"      media(A)={A.mean():.1f}  media(B)={B.mean():.1f}  "
          f"media(sat)={s_sat.mean():.1f}  media(mod)={s_mod.mean():.1f}  media(pond)={s_pon.mean():.1f}")
    reg("suma", "P2", "pct_desborde", 100 * frac_sat)
    for k, v in (("sat", s_sat), ("mod", s_mod), ("pond", s_pon)):
        reg("suma", "P2", f"media_{k}", v.mean())
        guardar_img(ims / f"suma_P2_{k}.png", v)

    fig, axs = plt.subplots(2, 3, figsize=(8, 5.6))
    mostrar(axs[0, 0], A, f"A = img{i}")
    mostrar(axs[0, 1], B, f"B = img{j}")
    mostrar(axs[0, 2], s_sat, f"A+B saturada  (desborda {100 * frac_sat:.1f} %)")
    mostrar(axs[1, 0], s_mod, "A+B módulo 256")
    mostrar(axs[1, 1], s_pon, "0.5·A + 0.5·B")
    hist_gris(axs[1, 2], s_sat, "saturada", PURPURA)
    hist_gris(axs[1, 2], s_mod, "módulo 256", "#e08400")
    hist_gris(axs[1, 2], s_pon, "ponderada", "#00a0a0")
    axs[1, 2].set_title("Histogramas (gris, eje Y log)", fontsize=8)
    axs[1, 2].legend(fontsize=7)
    axs[1, 2].tick_params(labelsize=7)
    cerrar(fig, figs / "fig04_suma_P2.png")

    # 2b) Suma de un escalar (brillo) con histogramas
    base = prep[3]
    ks = [0, 60, 120, 200]
    fig, axs = plt.subplots(3, 4, figsize=(8, 6.2))
    print("   Suma de escalar k sobre img3:")
    for c, k in enumerate(ks):
        o_s, o_m = suma_escalar_sat(base, k), suma_escalar_mod(base, k)
        p_s = 100 * np.mean(o_s == 255)
        p_m = 100 * np.mean((base.astype(np.uint16) + k) > 255)
        print(f"      k={k:>3}: valores en 255 (sat) = {p_s:5.2f} %   valores que dan la vuelta (mod) = {p_m:5.2f} %")
        reg("suma_escalar", f"k={k}", "pct_255_sat", p_s)
        reg("suma_escalar", f"k={k}", "pct_wrap_mod", p_m)
        mostrar(axs[0, c], o_s, f"saturada, k={k}  ({p_s:.1f}% en 255)")
        mostrar(axs[1, c], o_m, f"módulo 256, k={k}  ({p_m:.1f}% con wrap)")
        hist_gris(axs[2, c], o_s, "saturada", PURPURA, 0.9)
        axs[2, c].set_ylim(1, base.size)
        axs[2, c].set_xlim(0, 255)
        axs[2, c].set_title("hist. (saturada)", fontsize=8)
        axs[2, c].tick_params(labelsize=7)
        guardar_img(ims / f"suma_escalar_k{k}_sat.png", o_s)
        guardar_img(ims / f"suma_escalar_k{k}_mod.png", o_m)
    cerrar(fig, figs / "fig05_suma_escalar.png")

    # 2c) Par P3 + barrido de alfa
    i, j = PARES["P3"]
    A, B = prep[i], prep[j]
    fig, axs = plt.subplots(2, 4, figsize=(8, 4.6))
    mostrar(axs[0, 0], A, f"A = img{i}")
    mostrar(axs[0, 1], B, f"B = img{j}")
    mostrar(axs[0, 2], suma_saturada(A, B), "A+B saturada")
    mostrar(axs[0, 3], suma_modular(A, B), "A+B módulo 256")
    for c, al in enumerate((0.25, 0.5, 0.75)):
        o = suma_ponderada(A, B, al)
        mostrar(axs[1, c], o, f"α={al}:  α·A+(1−α)·B")
        guardar_img(ims / f"suma_P3_alfa{int(al * 100)}.png", o)
    ocultar_sobrantes(axs[1:], 3)
    axs[1, 3].axis("off")
    frac3 = np.mean((A.astype(np.uint16) + B.astype(np.uint16)) > 255)
    reg("suma", "P3", "pct_desborde", 100 * frac3)
    print(f"   P3 (img{i}+img{j}): valores que desbordan 255 = {100 * frac3:.2f} %")
    guardar_img(ims / "suma_P3_sat.png", suma_saturada(A, B))
    guardar_img(ims / "suma_P3_mod.png", suma_modular(A, B))
    cerrar(fig, figs / "fig06_suma_P3.png")

    # ------------------------------------------------------------------ #
    titulo("3. RESTA")
    # 3a) P1: efecto del registro
    i, j = PARES["P1"]
    A, B = prep[i], prep[j]
    Breg, W, cc = registrar_ecc(A, B)
    d_pre, d_post = diferencia_abs(A, B), diferencia_abs(A, Breg)
    print(f"   P1 (img{i} vs img{j}) registro ECC afín: cc={cc:.4f}")
    print(f"      matriz afín = {np.round(W, 4).tolist()}")
    print(f"      MAE sin registro = {mae(A, B):.3f}   con registro = {mae(A, Breg):.3f}")
    print(f"      PSNR sin registro = {psnr(A, B):.2f} dB   con registro = {psnr(A, Breg):.2f} dB")
    print(f"      píxeles con |A-B|>10: sin registro {100 * np.mean(d_pre > 10):.2f} %   "
          f"con registro {100 * np.mean(d_post > 10):.3f} %")
    for k, v in (("MAE_pre", mae(A, B)), ("MAE_post", mae(A, Breg)),
                 ("PSNR_pre", psnr(A, B)), ("PSNR_post", psnr(A, Breg)),
                 ("pct_gt10_pre", 100 * np.mean(d_pre > 10)),
                 ("pct_gt10_post", 100 * np.mean(d_post > 10)),
                 ("cc_ecc", cc), ("escala_afin", W[0, 0])):
        reg("resta", "P1", k, v)
    guardar_img(ims / "resta_P1_absdiff_sin_registro.png", np.clip(d_pre.astype(np.int32) * 4, 0, 255).astype(np.uint8))
    guardar_img(ims / "resta_P1_absdiff_con_registro.png", np.clip(d_post.astype(np.int32) * 20, 0, 255).astype(np.uint8))
    guardar_img(ims / "prep_img5_registrada.png", Breg)

    fig, axs = plt.subplots(2, 3, figsize=(8, 5.6))
    mostrar(axs[0, 0], A, f"A = img{i}")
    mostrar(axs[0, 1], B, f"B = img{j} (sin registrar)")
    mostrar(axs[0, 2], Breg, f"B registrada (ECC afín, cc={cc:.3f})")
    mostrar(axs[1, 0], np.clip(d_pre.astype(np.int32) * 4, 0, 255).astype(np.uint8),
            f"|A−B| sin registro ×4   (MAE {mae(A, B):.2f})")
    mostrar(axs[1, 1], np.clip(d_post.astype(np.int32) * 20, 0, 255).astype(np.uint8),
            f"|A−B_reg| con registro ×20   (MAE {mae(A, Breg):.2f})")
    hist_gris(axs[1, 2], gris(d_pre), "sin registro", "#e08400")
    hist_gris(axs[1, 2], gris(d_post), "con registro", PURPURA)
    axs[1, 2].set_title("Histograma de |A−B| (eje Y log)", fontsize=8)
    axs[1, 2].legend(fontsize=7)
    axs[1, 2].tick_params(labelsize=7)
    cerrar(fig, figs / "fig07_resta_P1_registro.png")

    # 3b) P2: direccional, modular y absoluta
    i, j = PARES["P2"]
    A, B = prep[i], prep[j]
    r_ab, r_ba = resta_saturada(A, B), resta_saturada(B, A)
    r_mod, r_abs = resta_modular(A, B), diferencia_abs(A, B)
    neg = 100 * np.mean(A.astype(np.int16) - B.astype(np.int16) < 0)
    print(f"   P2 (img{i}-img{j}): valores con A-B<0 = {neg:.2f} %  (se truncan a 0 en saturada; "
          f"dan la vuelta en módulo)")
    print(f"      media(A-B sat)={r_ab.mean():.2f}  media(B-A sat)={r_ba.mean():.2f}  "
          f"media(mod)={r_mod.mean():.2f}  media(|A-B|)={r_abs.mean():.2f}")
    for k, v in (("pct_negativos", neg), ("media_AmB_sat", r_ab.mean()),
                 ("media_BmA_sat", r_ba.mean()), ("media_mod", r_mod.mean()),
                 ("media_absdiff", r_abs.mean())):
        reg("resta", "P2", k, v)
    for k, v in (("AmB_sat", r_ab), ("BmA_sat", r_ba), ("mod", r_mod), ("absdiff", r_abs)):
        guardar_img(ims / f"resta_P2_{k}.png", v)
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.6))
    mostrar(axs[0, 0], A, f"A = img{i}")
    mostrar(axs[0, 1], B, f"B = img{j}")
    mostrar(axs[0, 2], r_abs, "|A−B|")
    mostrar(axs[1, 0], r_ab, "A−B saturada")
    mostrar(axs[1, 1], r_ba, "B−A saturada")
    mostrar(axs[1, 2], r_mod, "A−B módulo 256")
    cerrar(fig, figs / "fig08_resta_P2.png")

    # 3c) Resta como sustracción de fondo
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.6))
    for r, k in enumerate((4, 6)):
        L = cv2.cvtColor(prep[k], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        inv = 255.0 - L
        fino = cv2.GaussianBlur(inv, (0, 0), 3.0)
        fondo = cv2.GaussianBlur(inv, (0, 0), 35.0)
        resid = np.clip((fino - fondo) * 2.0 + 128, 0, 255).astype(np.uint8)
        mostrar(axs[r, 0], np.clip(fino, 0, 255).astype(np.uint8), f"img{k}: 255−L* (σ=3)")
        mostrar(axs[r, 1], np.clip(fondo, 0, 255).astype(np.uint8), "fondo estimado (σ=35)")
        mostrar(axs[r, 2], resid, "resta ×2 + 128 (fondo corregido)")
        guardar_img(ims / f"resta_fondo_img{k}.png", resid)
    cerrar(fig, figs / "fig09_resta_fondo.png")

    # ------------------------------------------------------------------ #
    titulo("4. AND")
    # 4a) AND directo entre imágenes
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.0))
    for r, (nom, (i, j)) in enumerate((("P2", PARES["P2"]), ("P3", PARES["P3"]))):
        A, B = prep[i], prep[j]
        o = op_and(A, B)
        print(f"   {nom}: media(A)={A.mean():6.1f}  media(B)={B.mean():6.1f}  media(A AND B)={o.mean():6.1f}  "
              f"(cota: <= min por píxel)")
        reg("and", nom, "media_A", A.mean())
        reg("and", nom, "media_B", B.mean())
        reg("and", nom, "media_out", o.mean())
        guardar_img(ims / f"and_{nom}.png", o)
        mostrar(axs[r, 0], A, f"A = img{i}")
        mostrar(axs[r, 1], B, f"B = img{j}")
        mostrar(axs[r, 2], o, f"A AND B  (media {o.mean():.0f})")
    cerrar(fig, figs / "fig10_and_directo.png")

    # 4b) AND con máscara constante: segmentación por planos de bits
    base = prep[3]
    mascaras = [(0xFF, 8), (0xFC, 6), (0xF0, 4), (0xC0, 2), (0x80, 1)]
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.4))
    psnrs = []
    print("   AND con máscara constante sobre img3 (bits más significativos conservados):")
    for ax, (m, nb) in zip(axs.ravel(), mascaras):
        o = op_and(base, np.full_like(base, m))
        p = psnr(base, o)
        psnrs.append(p)
        niveles = 2 ** nb
        print(f"      máscara 0x{m:02X}  ({nb} bit/canal, {niveles:>3} niveles)  "
              f"PSNR = {'∞' if np.isinf(p) else f'{p:5.2f} dB'}")
        reg("and_bitplano", f"0x{m:02X}", "PSNR", 99.0 if np.isinf(p) else p)
        guardar_img(ims / f"and_bitplano_0x{m:02X}.png", o)
        mostrar(ax, o, f"AND 0x{m:02X}  ({nb} bit, {niveles} niveles)")
    ax = axs[1, 2]
    finitos = [p for p in psnrs if not np.isinf(p)]
    ax.bar([f"{nb} b" for _, nb in mascaras[1:]], finitos, color=PURPURA)
    ax.set_title("PSNR frente al original (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    for x, v in enumerate(finitos):
        ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    cerrar(fig, figs / "fig11_and_bitplanos.png")

    # 4c) Máscaras de lesión: AND imagen-máscara (extracción de ROI)
    masks, resid = {}, {}
    for k in range(1, 8):
        masks[k], resid[k] = mascara_lesion(prep[k])
        cob = 100 * np.mean(masks[k] > 0)
        reg("mascara", f"img{k}", "cobertura_pct", cob)
    print("   Cobertura de la máscara (% del área):  " +
          "  ".join(f"img{k}={100 * np.mean(masks[k] > 0):.1f}" for k in range(1, 8)))

    fig, axs = plt.subplots(4, 3, figsize=(7.2, 9.4))
    for r, k in enumerate((4, 6, 3, 2)):
        roi = op_and(prep[k], cv2.merge([masks[k]] * 3))
        guardar_img(ims / f"and_roi_img{k}.png", roi)
        guardar_img(ims / f"mascara_img{k}.png", masks[k])
        mostrar(axs[r, 0], prep[k], f"img{k}")
        mostrar(axs[r, 1], masks[k], f"máscara M{k} (Otsu corregido)")
        mostrar(axs[r, 2], roi, f"img{k} AND M{k}")
    cerrar(fig, figs / "fig12_and_mascara_roi.png")

    # 4d) Casos donde el umbral único falla
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for r, k in enumerate((1, 7)):
        roi = op_and(prep[k], cv2.merge([masks[k]] * 3))
        guardar_img(ims / f"and_roi_img{k}.png", roi)
        guardar_img(ims / f"mascara_img{k}.png", masks[k])
        mostrar(axs[r, 0], prep[k], f"img{k}")
        mostrar(axs[r, 1], masks[k], f"máscara M{k}")
        mostrar(axs[r, 2], roi, f"img{k} AND M{k}")
    cerrar(fig, figs / "fig13_and_limitaciones.png")

    # 4e) Intersección de máscaras + IoU (AND / OR en conteo)
    Breg_mask, _ = mascara_lesion(Breg)
    pares_iou = [("img3-img5 sin registro", masks[3], masks[5]),
                 ("img3-img5 con registro", masks[3], Breg_mask),
                 ("img4-img6", masks[4], masks[6]),
                 ("img2-img4", masks[2], masks[4]),
                 ("img2-img6", masks[2], masks[6]),
                 ("img2-img3", masks[2], masks[3])]
    print("   IoU = |M_a AND M_b| / |M_a OR M_b|:")
    for nom, ma, mb in pares_iou:
        v = iou(ma, mb)
        print(f"      {nom:<24} IoU = {v:.3f}")
        reg("iou", nom, "IoU", v)

    # ------------------------------------------------------------------ #
    titulo("5. OR")
    # 5a) OR directo
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.0))
    for r, (nom, (i, j)) in enumerate((("P2", PARES["P2"]), ("P3", PARES["P3"]))):
        A, B = prep[i], prep[j]
        o = op_or(A, B)
        print(f"   {nom}: media(A)={A.mean():6.1f}  media(B)={B.mean():6.1f}  media(A OR B)={o.mean():6.1f}  "
              f"(cota: >= max por píxel)")
        reg("or", nom, "media_A", A.mean())
        reg("or", nom, "media_B", B.mean())
        reg("or", nom, "media_out", o.mean())
        guardar_img(ims / f"or_{nom}.png", o)
        mostrar(axs[r, 0], A, f"A = img{i}")
        mostrar(axs[r, 1], B, f"B = img{j}")
        mostrar(axs[r, 2], o, f"A OR B  (media {o.mean():.0f})")
    cerrar(fig, figs / "fig14_or_directo.png")

    # 5b) Unión de máscaras y fotomontaje (AND + OR)
    i, j = PARES["P2"]
    ma, mb = masks[i], masks[j]
    union, inter = op_or(ma, mb), op_and(ma, mb)
    lesion_a = op_and(prep[i], cv2.merge([ma] * 3))
    lesion_b = op_and(prep[j], cv2.merge([mb] * 3))
    montaje = op_or(lesion_a, lesion_b)
    solape = 100 * cv2.countNonZero(inter) / max(cv2.countNonZero(union), 1)
    print(f"   P2 máscaras: |Ma|={cv2.countNonZero(ma)}  |Mb|={cv2.countNonZero(mb)}  "
          f"|Ma AND Mb|={cv2.countNonZero(inter)}  |Ma OR Mb|={cv2.countNonZero(union)}  IoU={solape / 100:.3f}")
    # fondo blanco: (img AND M) OR (255 AND NOT M)
    blanco_a = op_or(lesion_a, cv2.merge([cv2.bitwise_not(ma)] * 3))
    guardar_img(ims / "or_union_mascaras_P2.png", union)
    guardar_img(ims / "or_montaje_P2.png", montaje)
    guardar_img(ims / f"or_fondo_blanco_img{i}.png", blanco_a)
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.6))
    mostrar(axs[0, 0], ma, f"M{i}")
    mostrar(axs[0, 1], mb, f"M{j}")
    mostrar(axs[0, 2], union, f"M{i} OR M{j}  (unión)")
    mostrar(axs[1, 0], lesion_a, f"img{i} AND M{i}")
    mostrar(axs[1, 1], lesion_b, f"img{j} AND M{j}")
    mostrar(axs[1, 2], montaje, "(img4 AND M4) OR (img6 AND M6)")
    cerrar(fig, figs / "fig15_or_mascaras.png")

    # ------------------------------------------------------------------ #
    titulo("6. AND + OR COMBINADOS: ocultar una imagen en los 4 bits bajos de otra")
    portadora, secreta = prep[PARES["P2"][0]], prep[PARES["P2"][1]]
    stego = op_or(op_and(portadora, np.full_like(portadora, 0xF0)),
                  (secreta >> 4).astype(np.uint8))
    recup = (op_and(stego, np.full_like(stego, 0x0F)) << 4).astype(np.uint8)
    assert np.array_equal(recup, op_and(secreta, np.full_like(secreta, 0xF0)))
    p_port, p_sec = psnr(portadora, stego), psnr(secreta, recup)
    print(f"   PSNR(portadora, stego) = {p_port:.2f} dB   PSNR(secreta, recuperada) = {p_sec:.2f} dB")
    print("   Comprobado: recuperada == (secreta AND 0xF0)  -> True")
    reg("combinado", "stego", "PSNR_portadora", p_port)
    reg("combinado", "stego", "PSNR_secreta", p_sec)
    guardar_img(ims / "combinado_stego.png", stego)
    guardar_img(ims / "combinado_recuperada.png", recup)
    fig, axs = plt.subplots(2, 3, figsize=(8, 5.6))
    mostrar(axs[0, 0], portadora, "portadora  P = img4")
    mostrar(axs[0, 1], secreta, "secreta  S = img6")
    mostrar(axs[0, 2], stego, f"(P AND 0xF0) OR (S>>4)   {p_port:.1f} dB")
    mostrar(axs[1, 0], recup, f"(stego AND 0x0F)<<4   {p_sec:.1f} dB")
    mostrar(axs[1, 1], np.clip(diferencia_abs(portadora, stego).astype(np.int32) * 16, 0, 255).astype(np.uint8),
            "|P − stego| ×16")
    mostrar(axs[1, 2], np.clip(diferencia_abs(secreta, recup).astype(np.int32) * 16, 0, 255).astype(np.uint8),
            "|S − recuperada| ×16")
    cerrar(fig, figs / "fig16_combinado.png")

    # ------------------------------------------------------------------ #
    args.salida.mkdir(parents=True, exist_ok=True)
    with open(args.salida / "metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["operacion", "contexto", "metrica", "valor"])
        w.writerows(METRICAS)
    titulo(f"LISTO  ({time.perf_counter() - t0:.1f} s)  ->  {args.salida.resolve()}")
    print(f"   {len(list(figs.glob('*.png')))} figuras, {len(list(ims.glob('*.png')))} imágenes, "
          f"{len(METRICAS)} métricas en metricas.csv")


if __name__ == "__main__":
    main()
