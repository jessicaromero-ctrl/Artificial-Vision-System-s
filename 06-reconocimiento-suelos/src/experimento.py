"""Orquestación de los dos experimentos y generación de salidas (figuras y JSON)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support  # noqa: E402

from . import datos, evaluacion  # noqa: E402
from .caracteristicas import ExtractorHaralick, ExtractorPixeles  # noqa: E402

NOMBRES = {"sin_haralick": "Sin Haralick (píxeles)", "con_haralick": "Con Haralick (GLCM)"}


def ejecutar(raiz: Path, salida: Path, lado: int, semillas, n_boot: int, n_perm: int, n_jobs: int,
             limite_por_clase: int | None = None) -> dict:
    fig_dir = salida / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"semillas": list(semillas), "lado_imagen": lado}

    # 1. Datos y control de calidad
    filas = datos.catalogo(raiz)  # train y test unidos
    rep["imagenes_brutas"] = len(filas)
    filas, dup = datos.quitar_duplicados_exactos(filas)
    rep.update(dup)
    if limite_por_clase:
        rng = np.random.default_rng(0)
        clases = np.array([f[1] for f in filas])
        sel = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(clases == c), size=min(limite_por_clase, int((clases == c).sum())),
                       replace=False) for c in np.unique(clases)]))
        filas = [filas[i] for i in sel]
    clases = sorted({f[1] for f in filas})
    y = np.array([clases.index(f[1]) for f in filas])
    rep["clases"] = clases
    rep["imagenes_finales"] = len(filas)
    rep["imagenes_por_clase"] = {c: int((y == i).sum()) for i, c in enumerate(clases)}

    # 2. Preprocesamiento y características
    t0 = time.perf_counter()
    grises = datos.cargar_grises([f[0] for f in filas], lado)
    t_carga = time.perf_counter() - t0
    extractores = {"sin_haralick": ExtractorPixeles(), "con_haralick": ExtractorHaralick()}
    X, t_ext = {}, {}
    for k, ext in extractores.items():
        t0 = time.perf_counter()
        X[k] = ext.extraer(grises)
        t_ext[k] = time.perf_counter() - t0
    rep["costo_extraccion"] = {"carga_s": t_carga,
                               **{f"{k}_s": v for k, v in t_ext.items()},
                               "haralick_ms_por_imagen": 1000 * t_ext["con_haralick"] / len(filas),
                               "dimension": {k: int(v.shape[1]) for k, v in X.items()}}

    hashes = np.array([datos.ahash(g) for g in grises], dtype=np.uint64)
    pares = datos.pares_casi_duplicados(hashes)
    rep["casi_duplicados"] = {"umbral_hamming": datos.AHASH_UMBRAL, "pares": len(pares),
                              "pares_misma_clase": int(sum(y[i] == y[j] for i, j in pares))}

    # 3. Experimentos por semilla
    resultados = {k: [] for k in NOMBRES}
    for k in NOMBRES:
        for s in semillas:
            r = evaluacion.ejecutar_particion(X[k], y, s, con_pca=(k == "sin_haralick"), n_jobs=n_jobs)
            r["ic95"] = evaluacion.ic_bootstrap(r["aciertos"], n_boot, s)
            resultados[k].append(r)
            print(f"[{k}] semilla={s} exactitud={r['exactitud']:.4f} "
                  f"IC95=[{r['ic95'][0]:.4f}, {r['ic95'][1]:.4f}] C={r['C']} cv={r['cv_media']:.4f}")

    # Casi-duplicados cruzados entre entrenamiento y prueba (semilla base)
    r0 = resultados["sin_haralick"][0]
    tr_s, te_s = set(r0["tr"].tolist()), set(r0["te"].tolist())
    rep["casi_duplicados"]["pares_cruzados_entrenamiento_prueba"] = int(
        sum((i in tr_s and j in te_s) or (j in tr_s and i in te_s) for i, j in pares))

    # 4. Resumen con IC t sobre semillas
    resumen = {}
    for k in NOMBRES:
        ex = np.array([r["exactitud"] for r in resultados[k]])
        m, lo, hi = evaluacion.ic_t(ex)
        resumen[k] = {
            "nombre": NOMBRES[k], "exactitud_media": m, "ic95_t_inf": lo, "ic95_t_sup": hi,
            "desv_std": float(ex.std(ddof=1)),
            "f1_macro_media": float(np.mean([r["f1_macro"] for r in resultados[k]])),
            "cv_media": float(np.mean([r["cv_media"] for r in resultados[k]])),
            "t_ajuste_media_s": float(np.mean([r["t_ajuste_s"] for r in resultados[k]])),
            "t_inferencia_ms_media": float(np.mean([r["t_inferencia_ms"] for r in resultados[k]])),
            "por_semilla": [{"semilla": r["semilla"], "exactitud": r["exactitud"], "ic95": r["ic95"],
                             "C": r["C"], "cv_media": r["cv_media"], "cv_desv": r["cv_desv"]}
                            for r in resultados[k]],
        }
    rep["resumen"] = resumen

    # 5. Comparación pareada
    a, b = resultados["sin_haralick"], resultados["con_haralick"]
    dif = np.array([rb["exactitud"] - ra["exactitud"] for ra, rb in zip(a, b)])
    t_st, t_p = stats.ttest_rel([r["exactitud"] for r in b], [r["exactitud"] for r in a])
    dm, dlo, dhi = evaluacion.ic_t(dif)
    rep["comparacion"] = {
        "diferencia_media_con_menos_sin": dm, "ic95_diferencia": [dlo, dhi],
        "t_pareada_semillas": {"t": float(t_st), "p": float(t_p), "gl": len(dif) - 1},
        "mcnemar_por_semilla": [evaluacion.mcnemar(ra["aciertos"], rb["aciertos"]) for ra, rb in zip(a, b)],
    }

    # 6. Permutación (semilla base) y figuras
    rep["permutacion"] = {}
    for k in NOMBRES:
        r = resultados[k][0]
        nula, p = evaluacion.permutacion(X[k], y, r["tr"], r["te"], r["C"], r["exactitud"],
                                         n_perm, semillas[0], con_pca=(k == "sin_haralick"))
        rep["permutacion"][k] = {"exactitud_observada": r["exactitud"], "nula_media": float(nula.mean()),
                                 "nula_p95": float(np.percentile(nula, 95)), "p_valor": p,
                                 "n_permutaciones": n_perm}
        _graficar_permutacion(nula, r["exactitud"], NOMBRES[k], fig_dir / f"permutacion_{k}.png")
        _graficar_confusion(r["y_prueba"], r["pred"], clases, NOMBRES[k], fig_dir / f"confusion_{k}.png")

    # 7. F1 por clase
    filas_f1 = []
    for k in NOMBRES:
        F = np.vstack([precision_recall_fscore_support(r["y_prueba"], r["pred"],
                                                       labels=range(len(clases)), zero_division=0)[2]
                       for r in resultados[k]])
        for i, c in enumerate(clases):
            filas_f1.append({"experimento": NOMBRES[k], "clase": c,
                             "f1_media": F[:, i].mean(), "f1_desv": F[:, i].std(ddof=1)})
    pd.DataFrame(filas_f1).to_csv(salida / "f1_por_clase.csv", index=False)

    _graficar_muestra(grises, [f[1] for f in filas], clases, fig_dir / "muestra_por_clase.png")
    _graficar_ic(resumen, fig_dir / "exactitud_ic95.png")
    with open(salida / "resultados.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=float)
    return rep


def _graficar_permutacion(nula, observada, titulo, ruta):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(nula, bins=20, color="#c9a2f0", edgecolor="white")
    ax.axvline(observada, color="#8400e0", lw=2, label=f"observada = {observada:.3f}")
    ax.set_xlabel("Exactitud en prueba (etiquetas permutadas)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _graficar_confusion(y_true, y_pred, clases, titulo, ruta):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(clases)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _graficar_muestra(grises, etiquetas, clases, ruta):
    fig, ejes = plt.subplots(1, len(clases), figsize=(2.3 * len(clases), 2.6))
    for ax, c in zip(np.atleast_1d(ejes), clases):
        ax.imshow(grises[etiquetas.index(c)], cmap="gray")
        ax.set_title(c, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _graficar_ic(resumen, ruta):
    nombres = [v["nombre"] for v in resumen.values()]
    m = np.array([v["exactitud_media"] for v in resumen.values()])
    lo = np.array([v["ic95_t_inf"] for v in resumen.values()])
    hi = np.array([v["ic95_t_sup"] for v in resumen.values()])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(nombres, m, xerr=np.vstack([m - lo, hi - m]), color=["#c9a2f0", "#8400e0"], capsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Exactitud media en prueba (IC 95 % t sobre semillas)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
