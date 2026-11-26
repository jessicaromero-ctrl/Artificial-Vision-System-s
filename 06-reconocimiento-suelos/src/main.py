"""Punto de entrada: reproduce los dos experimentos.

    python -m src.main --datos RUTA_DEL_DATASET [--salida salidas]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import experimento

RAIZ = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, required=True, help="carpeta del dataset (kagglehub o zip descomprimido)")
    ap.add_argument("--salida", type=Path, default=RAIZ / "salidas")
    ap.add_argument("--lado", type=int, default=128)
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limite-por-clase", type=int, default=None, help="solo pruebas rápidas")
    a = ap.parse_args(argv)
    rep = experimento.ejecutar(a.datos, a.salida, a.lado, a.semillas, a.n_boot, a.n_perm,
                               a.n_jobs, a.limite_por_clase)
    print(f"Listo: {rep['imagenes_finales']} imágenes; resultados en {a.salida}")


if __name__ == "__main__":
    main()
