"""Extractores de características (patrón Estrategia).

    ExtractorPixeles  -> experimento SIN Haralick: píxeles reducidos y normalizados.
    ExtractorHaralick -> experimento CON Haralick: GLCM sobre los píxeles de la imagen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from . import haralick


class Extractor(ABC):
    nombre: str

    @abstractmethod
    def extraer(self, grises: np.ndarray) -> np.ndarray:
        """grises: (N, L, L) uint8 -> matriz (N, d) de características."""


class ExtractorPixeles(Extractor):
    nombre = "sin_haralick"

    def __init__(self, lado: int = 64):
        self.lado = lado

    def extraer(self, grises):
        salida = np.empty((len(grises), self.lado * self.lado), dtype=np.float32)
        for k, g in enumerate(grises):
            pequena = Image.fromarray(g).resize((self.lado, self.lado), Image.BILINEAR)
            salida[k] = np.asarray(pequena, dtype=np.float32).ravel() / 255.0
        return salida


class ExtractorHaralick(Extractor):
    nombre = "con_haralick"

    def __init__(self, distancias=(1, 2, 4), niveles: int = 32):
        self.distancias = tuple(distancias)
        self.niveles = niveles

    @property
    def dimension(self) -> int:
        return len(self.distancias) * 13 * 2

    def extraer(self, grises):
        return np.vstack([haralick.vector_haralick(g, self.distancias, self.niveles)
                          for g in grises])
