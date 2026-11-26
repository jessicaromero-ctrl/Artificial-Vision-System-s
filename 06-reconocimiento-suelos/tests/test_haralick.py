"""Valida la implementación propia de Haralick contra mahotas (referencia externa)."""
import sys
from pathlib import Path

import mahotas
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.haralick import DIRECCIONES, NOMBRES, cuantizar, glcm, descriptores, haralick_direcciones  # noqa: E402

NIVELES = 32


def _imagen(seed=0, shape=(96, 96)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint16).astype(np.uint8)


def test_forma_y_normalizacion():
    q = cuantizar(_imagen(), NIVELES)
    P = glcm(q, 1, (0, 1), NIVELES)
    assert P.shape == (NIVELES, NIVELES)
    assert np.isclose(P.sum(), 1.0)
    assert np.allclose(P, P.T)


def test_coincide_con_mahotas_por_direccion():
    gris = _imagen(seed=1)
    q = cuantizar(gris, NIVELES)
    for d in (1, 2, 4):
        ref = mahotas.features.haralick(q.astype(np.uint8), distance=d, return_mean=False,
                                        ignore_zeros=False)
        ours = haralick_direcciones(q, d, NIVELES)
        assert ref.shape == ours.shape == (4, 13)
        for k, nombre in enumerate(NOMBRES):
            # Las orientaciones de mahotas están en otro orden: se compara el conjunto
            # ordenado de los 4 valores (suficiente, pues el vector usa media y rango).
            ok = np.allclose(np.sort(ours[:, k]), np.sort(ref[:, k]), rtol=1e-6, atol=1e-8)
            assert ok, f"distancia {d}, descriptor {nombre}: propio={ours[:, k]} mahotas={ref[:, k]}"


if __name__ == "__main__":
    test_forma_y_normalizacion()
    test_coincide_con_mahotas_por_direccion()
    print("OK: pruebas de Haralick pasaron")
