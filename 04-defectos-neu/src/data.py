"""Descubrimiento, verificación y particionado del dataset NEU de defectos superficiales.

Supuesto: cada imagen vive en una carpeta cuyo nombre es la clase del defecto
(p. ej. .../train/crazing/img_001.jpg). El particionado se hace aquí con semilla
fija, para no depender de cómo Kaggle organice las carpetas train/valid.
"""

import hashlib
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_CLASSES = 6  # crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches


def discover_samples(root):
    """Devuelve una lista de (ruta, clase) recorriendo recursivamente `root`."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {root}")

    samples = [
        (p, p.parent.name)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    if not samples:
        raise FileNotFoundError(f"No se encontraron imágenes en {root}")

    classes = sorted({label for _, label in samples})
    if len(classes) != EXPECTED_CLASSES:
        warnings.warn(
            f"Se detectaron {len(classes)} clases ({classes}); "
            f"NEU tiene {EXPECTED_CLASSES}. Revisa la estructura de carpetas."
        )
    return samples, classes


def deduplicate(samples):
    """Elimina archivos con contenido idéntico (hash MD5), conservando la primera aparición.

    Si una misma imagen queda en train y en prueba, las métricas están infladas
    (fuga de datos). Esta función se aplica antes del particionado.
    El dataset original en disco no se modifica; solo se excluyen copias en memoria.

    Devuelve (muestras_unicas, numero_de_duplicados_eliminados).
    """
    seen = set()
    unique, removed = [], []
    for path, label in samples:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            removed.append(path)
        else:
            seen.add(digest)
            unique.append((path, label))
    return unique, removed


def stratified_split(samples, test_size=0.2, seed=42):
    """Particiona manteniendo la proporción de clases en cada subconjunto."""
    paths = [p for p, _ in samples]
    labels = [l for _, l in samples]
    tr_p, te_p, tr_l, te_l = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return list(zip(tr_p, tr_l)), list(zip(te_p, te_l))


def check_leakage(train_samples, test_samples):
    """Devuelve las rutas de prueba cuyo contenido (hash MD5) también aparece en entrenamiento.

    Lista vacía significa que no hay fuga de datos entre particiones.
    """
    train_hashes = {hashlib.md5(p.read_bytes()).hexdigest() for p, _ in train_samples}
    return [p for p, _ in test_samples if hashlib.md5(p.read_bytes()).hexdigest() in train_hashes]
