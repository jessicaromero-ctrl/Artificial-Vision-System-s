"""Intervalos de confianza por bootstrap y pruebas pareadas entre modelos.

Notación:
    y_true, y_pred : etiquetas reales y predichas (arreglos de longitud n).
    A, B           : dos modelos evaluados sobre el MISMO conjunto de prueba.

Los intervalos de confianza cuantifican la incertidumbre por tamaño finito de la
prueba. La prueba de McNemar compara dos clasificadores sobre casos pareados.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def _metric(metric, y_true, y_pred, labels):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    raise ValueError(f"Métrica no soportada: {metric}")


def bootstrap_ci(y_true, y_pred, metric="accuracy", labels=None,
                 n_boot=2000, alpha=0.05, seed=42):
    """Intervalo de confianza percentil de (1 - alpha) por bootstrap.

    Se remuestrean n casos con reemplazo, n_boot veces, y se toman los percentiles
    alpha/2 y 1 - alpha/2 de la distribución resultante.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = _metric(metric, y_true[idx], y_pred[idx], labels)
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "estimate": float(_metric(metric, y_true, y_pred, labels)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 1 - alpha,
        "n_boot": n_boot,
    }


def mcnemar_exact(y_true, pred_a, pred_b):
    """Prueba exacta de McNemar sobre casos pareados.

    Sea n01 = #(A acierta y B falla) y n10 = #(A falla y B acierta).
    Bajo H0 (ambos modelos tienen la misma tasa de error), n01 ~ Binomial(n01 + n10, 0.5).
    p = 2 * P(X <= min(n01, n10)) con X ~ Binomial(n, 0.5), acotado a 1.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    p_value = 1.0 if n == 0 else float(
        binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue
    )
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "p_value": p_value}


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=2000, alpha=0.05, seed=42):
    """IC bootstrap de la diferencia de exactitud  Δ = acc(A) - acc(B).

    Se remuestrean los mismos índices para ambos modelos, para respetar el pareamiento.
    Si el intervalo no contiene 0, la diferencia es significativa al nivel alpha.
    """
    y_true = np.asarray(y_true)
    a_ok = (np.asarray(pred_a) == y_true).astype(float)
    b_ok = (np.asarray(pred_b) == y_true).astype(float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = a_ok[idx].mean() - b_ok[idx].mean()
    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff_estimate": float(a_ok.mean() - b_ok.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 1 - alpha,
        "n_boot": n_boot,
    }
