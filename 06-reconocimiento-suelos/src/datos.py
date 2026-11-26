"""Catálogo, lectura y control de calidad del Soil DataSet.

Acepta la carpeta del dataset descargada con kagglehub o la versión reorganizada.
La clase es el nombre de la carpeta padre de cada imagen (sin la palabra "Soil").
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AHASH_UMBRAL = 4  # distancia de Hamming máxima para considerar casi-duplicado


def _clase(nombre_carpeta: str) -> str:
    """'Alluvial Soil' -> 'alluvial'; conserva el nombre si no contiene 'soil'."""
    limpio = re.sub(r"\bsoil\b", "", nombre_carpeta, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", limpio).strip().lower() or nombre_carpeta.lower()


def catalogo(raiz: Path):
    """Lista (ruta, clase) de todas las imágenes bajo raiz, uniendo train y test.

    Acepta la estructura original del dataset (p. ej. 'Soil Train/Soil Train/Alluvial Soil/')
    y la reorganizada (train/alluvial/). La clase es la carpeta padre de cada imagen.
    """
    filas = []
    for f in sorted(raiz.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in EXTENSIONES:
            continue
        if not any(part.lower().startswith(("soil train", "soil test", "train", "test"))
                   for part in f.relative_to(raiz).parts[:-1]):
            continue  # fuera de train/test, se ignora
        filas.append((f, _clase(f.parent.name)))
    if not filas:
        raise FileNotFoundError(f"Sin imágenes de train/test bajo {raiz}")
    return filas


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def quitar_duplicados_exactos(filas):
    """Conserva la primera aparición de cada contenido idéntico (SHA-256)."""
    vistos: dict[str, int] = {}
    conservadas, eliminadas, conflictivas = [], 0, 0
    for fila in filas:
        h = sha256(fila[0])
        if h in vistos:
            eliminadas += 1
            conflictivas += int(filas[vistos[h]][1] != fila[1])
            continue
        vistos[h] = len(conservadas)
        conservadas.append(fila)
    return conservadas, {"duplicados_exactos": eliminadas,
                         "duplicados_con_clase_distinta": conflictivas}


def cargar_grises(rutas, lado: int) -> np.ndarray:
    """Matriz (N, lado, lado) uint8 en escala de grises."""
    salida = np.empty((len(rutas), lado, lado), dtype=np.uint8)
    for k, r in enumerate(rutas):
        with Image.open(r) as im:
            salida[k] = np.asarray(im.convert("L").resize((lado, lado), Image.BILINEAR), dtype=np.uint8)
    return salida


def ahash(gris: np.ndarray) -> np.uint64:
    """Hash perceptual promedio de 64 bits (gradiente horizontal sobre 9x8)."""
    pequena = np.asarray(Image.fromarray(gris).resize((9, 8), Image.BILINEAR), dtype=np.float32)
    bits = np.packbits((pequena[:, 1:] > pequena[:, :-1]).ravel())
    return np.frombuffer(bits.tobytes(), dtype=np.uint64)[0]


def pares_casi_duplicados(hashes: np.ndarray, umbral: int = AHASH_UMBRAL, bloque: int = 512):
    """Pares (i, j), i < j, con distancia de Hamming <= umbral."""
    pares = []
    for ini in range(0, len(hashes), bloque):
        x = hashes[ini:ini + bloque, None] ^ hashes[None, :]
        r, c = np.nonzero(np.bitwise_count(x) <= umbral)
        r = r + ini
        m = r < c
        pares.extend(zip(r[m].tolist(), c[m].tolist()))
    return pares
