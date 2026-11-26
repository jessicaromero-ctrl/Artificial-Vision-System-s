"""Clasificadores y pruebas estadísticas del protocolo.

Protocolo por semilla s:
  1. Partición estratificada 80/20 (entrenamiento / prueba).
  2. Ajuste de C por validación cruzada de 5 pliegues SOLO en el entrenamiento.
  3. Evaluación única sobre la prueba.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

C_GRID = [0.1, 1.0, 10.0, 100.0]
TAMANO_PRUEBA = 0.2
PCA_VARIANZA = 0.95  # componentes necesarias para conservar 95 % de la varianza


def estimador(con_pca: bool) -> Pipeline:
    pasos = [("escala", StandardScaler())]
    if con_pca:
        pasos.append(("pca", PCA(n_components=PCA_VARIANZA, svd_solver="full")))
    pasos.append(("svc", SVC(kernel="rbf", gamma="scale")))
    return Pipeline(pasos)


def ejecutar_particion(X, y, semilla: int, con_pca: bool, n_jobs: int = -1) -> dict:
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TAMANO_PRUEBA, stratify=y, random_state=semilla)
    busqueda = GridSearchCV(estimador(con_pca), {"svc__C": C_GRID},
                            cv=StratifiedKFold(5, shuffle=True, random_state=semilla),
                            scoring="accuracy", n_jobs=n_jobs)
    t0 = time.perf_counter()
    busqueda.fit(X[tr], y[tr])
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = busqueda.predict(X[te])
    t_inf = time.perf_counter() - t0
    b = busqueda.best_index_
    return {
        "semilla": semilla, "tr": tr, "te": te, "pred": pred, "y_prueba": y[te],
        "aciertos": (pred == y[te]).astype(np.float64),
        "C": float(busqueda.best_params_["svc__C"]),
        "cv_media": float(busqueda.cv_results_["mean_test_score"][b]),
        "cv_desv": float(busqueda.cv_results_["std_test_score"][b]),
        "exactitud": float(accuracy_score(y[te], pred)),
        "f1_macro": float(f1_score(y[te], pred, average="macro")),
        "t_ajuste_s": t_ajuste,
        "t_inferencia_ms": 1000.0 * t_inf / len(te),
    }


def ic_bootstrap(aciertos: np.ndarray, n_boot: int, semilla: int):
    """IC 95 % percentil bootstrap de la exactitud (remuestreo de muestras de prueba)."""
    rng = np.random.default_rng(semilla)
    n = len(aciertos)
    medias = aciertos[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return tuple(float(v) for v in np.percentile(medias, [2.5, 97.5]))


def permutacion(X, y, tr, te, C: float, observada: float, n_perm: int, semilla: int, con_pca: bool):
    """Prueba de permutación: entrena con etiquetas barajadas; la prueba queda intacta.

    p = (1 + #{nula >= observada}) / (1 + n_perm). Si la exactitud observada no supera
    la distribución nula, no hay evidencia de que el resultado sea fruto de aprendizaje.
    """
    rng = np.random.default_rng(semilla)
    nula = np.empty(n_perm)
    for b in range(n_perm):
        m = estimador(con_pca).set_params(svc__C=C)
        m.fit(X[tr], rng.permutation(y[tr]))
        nula[b] = accuracy_score(y[te], m.predict(X[te]))
    p = (1 + np.sum(nula >= observada)) / (n_perm + 1)
    return nula, float(p)


def mcnemar(aciertos_a: np.ndarray, aciertos_b: np.ndarray) -> dict:
    a = aciertos_a.astype(bool)
    b_ = aciertos_b.astype(bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "estadistico": 0.0, "p": 1.0, "metodo": "sin discordancias"}
    if n < 25:
        return {"b": b, "c": c, "estadistico": float("nan"),
                "p": float(stats.binomtest(min(b, c), n, 0.5).pvalue), "metodo": "binomial exacta"}
    chi = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "estadistico": float(chi), "p": float(stats.chi2.sf(chi, 1)),
            "metodo": "chi-cuadrado con corrección de continuidad"}


def ic_t(valores: np.ndarray, nivel: float = 0.95):
    n = len(valores)
    m = float(valores.mean())
    h = stats.t.ppf(0.5 + nivel / 2, df=n - 1) * valores.std(ddof=1) / np.sqrt(n)
    return m, m - float(h), m + float(h)
