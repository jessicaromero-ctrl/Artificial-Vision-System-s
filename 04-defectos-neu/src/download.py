"""Descarga el dataset NEU con kagglehub.

Requisito: credenciales de Kaggle en ~/.kaggle/kaggle.json (o variables KAGGLE_USERNAME
y KAGGLE_KEY definidas en la terminal). Nunca pegues el token en archivos del repositorio.

Verifica el slug exacto del dataset en kaggle.com antes de ejecutar; el valor por defecto
puede cambiar si el autor lo reorganiza.
"""

import argparse

import kagglehub

DEFAULT_SLUG = "kaustubhdikshit/neu-surface-defect-database"


def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset NEU desde Kaggle.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Identificador del dataset en Kaggle.")
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.slug)
    print(f"Dataset descargado en: {path}")


if __name__ == "__main__":
    main()
