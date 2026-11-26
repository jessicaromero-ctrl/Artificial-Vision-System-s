"""Descriptores de textura de Haralick (1973) implementados en NumPy.

Notación:
    q      imagen cuantizada, q[y, x] en {0, ..., L-1}
    P(i,j) matriz de co-ocurrencia simétrica normalizada (GLCM)
    p_x, p_y       marginales por filas y columnas
    p_{x+y}(k)     distribución de la suma i + j = k
    p_{x-y}(k)     distribución de la diferencia i - j = k
"""
from __future__ import annotations

import numpy as np

# Desplazamientos (dy, dx) para 0°, 45°, 90° y 135°, multiplicados por la distancia.
DIRECCIONES = ((0, 1), (-1, 1), (-1, 0), (-1, -1))

NOMBRES = (
    "asm", "contraste", "correlacion", "varianza", "momento_inverso_diferencia",
    "promedio_suma", "varianza_suma", "entropia_suma", "entropia",
    "varianza_diferencia", "entropia_diferencia", "medida_info_corr_1", "medida_info_corr_2",
)
EPS = 1e-12


def _pares(q: np.ndarray, dy: int, dx: int):
    """Pares (valor en p, valor en p + (dy, dx)) para todos los píxeles válidos."""
    h, w = q.shape
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    a = q[y0:y1, x0:x1].ravel()
    b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
    return a, b


def glcm(q: np.ndarray, distancia: int, direccion: tuple[int, int], niveles: int) -> np.ndarray:
    """GLCM simétrica y normalizada para una distancia y una orientación."""
    dy, dx = direccion[0] * distancia, direccion[1] * distancia
    a, b = _pares(q, dy, dx)
    c = np.bincount(a.astype(np.int64) * niveles + b.astype(np.int64),
                    minlength=niveles * niveles).reshape(niveles, niveles).astype(np.float64)
    c = c + c.T
    total = c.sum()
    return c / total if total > 0 else c


def _xlogx(p: np.ndarray) -> np.ndarray:
    """p * log2(p), con 0 log 0 = 0. Las entropías usan base 2 (convención de mahotas)."""
    return np.where(p > 0, p * np.log2(np.maximum(p, EPS)), 0.0)


def descriptores(P: np.ndarray) -> np.ndarray:
    """Los 13 descriptores de Haralick de una GLCM normalizada, en el orden de NOMBRES."""
    L = P.shape[0]
    idx = np.arange(L, dtype=np.float64)
    i = idx[:, None]
    j = idx[None, :]
    px = P.sum(axis=1)
    py = P.sum(axis=0)
    mu_x = float(np.sum(idx * px))
    mu_y = float(np.sum(idx * py))
    sx = np.sqrt(np.sum((idx - mu_x) ** 2 * px))
    sy = np.sqrt(np.sum((idx - mu_y) ** 2 * py))

    # 1. Segundo momento angular (energía)
    asm = np.sum(P ** 2)
    # 2. Contraste
    contraste = np.sum((i - j) ** 2 * P)
    # 3. Correlación
    correlacion = (np.sum(i * j * P) - mu_x * mu_y) / (sx * sy + EPS)
    # 4. Varianza
    varianza = np.sum((i - mu_x) ** 2 * P)
    # 5. Momento inverso de la diferencia (homogeneidad)
    idm = np.sum(P / (1.0 + (i - j) ** 2))

    # Distribuciones de suma y diferencia
    k_suma = np.arange(2 * L - 1, dtype=np.float64)
    p_suma = np.bincount((i + j).astype(np.int64).ravel(), weights=P.ravel(), minlength=2 * L - 1)
    # Distribución de la diferencia plegada: p_{x-y}(k) = sum_{|i-j| = k} P(i, j), k = 0..L-1
    k_dif = np.arange(L, dtype=np.float64)
    p_dif = np.bincount(np.abs(i - j).astype(np.int64).ravel(), weights=P.ravel(), minlength=L)
    # 6. Promedio de la suma
    promedio_suma = np.sum(k_suma * p_suma)
    # 7. Varianza de la suma
    varianza_suma = np.sum((k_suma - promedio_suma) ** 2 * p_suma)
    # 8. Entropía de la suma
    entropia_suma = -np.sum(_xlogx(p_suma))
    # 9. Entropía
    entropia = -np.sum(_xlogx(P))
    # 10. Varianza de la diferencia
    # Convención de mahotas: varianza de los valores de p_{x-y} (no ponderada por k)
    varianza_diferencia = np.var(p_dif)
    # 11. Entropía de la diferencia
    entropia_diferencia = -np.sum(_xlogx(p_dif))

    # 12 y 13. Medidas de información de la correlación
    hx = -np.sum(_xlogx(px))
    hy = -np.sum(_xlogx(py))
    hxy = entropia
    outer = np.outer(px, py)
    hxy1 = -np.sum(P * np.log2(outer + EPS))
    hxy2 = -np.sum(outer * np.log2(outer + EPS))
    denom = max(hx, hy)
    imc1 = (hxy - hxy1) / denom if denom > EPS else 0.0
    imc2 = np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - hxy))))

    return np.array([asm, contraste, correlacion, varianza, idm, promedio_suma, varianza_suma,
                     entropia_suma, entropia, varianza_diferencia, entropia_diferencia,
                     imc1, imc2], dtype=np.float64)


def haralick_direcciones(q: np.ndarray, distancia: int, niveles: int) -> np.ndarray:
    """Matriz (4, 13): descriptores para cada orientación a una distancia dada."""
    return np.vstack([descriptores(glcm(q, distancia, d, niveles)) for d in DIRECCIONES])


def cuantizar(gris: np.ndarray, niveles: int) -> np.ndarray:
    """Reduce 256 niveles de gris a `niveles` valores enteros 0..niveles-1."""
    return (gris.astype(np.uint16) * niveles // 256).astype(np.int64)


def vector_haralick(gris: np.ndarray, distancias: tuple[int, ...], niveles: int) -> np.ndarray:
    """Vector de características: media y rango sobre orientaciones, por distancia.

    Dimensión: len(distancias) * 13 * 2.
    """
    q = cuantizar(gris, niveles)
    partes = []
    for d in distancias:
        m = haralick_direcciones(q, d, niveles)
        partes.append(m.mean(axis=0))
        partes.append(m.max(axis=0) - m.min(axis=0))
    return np.nan_to_num(np.concatenate(partes), nan=0.0, posinf=0.0, neginf=0.0)
