"""
Experimento: detección de esquinas con el detector de Harris.

Barrido del umbral absoluto δ sobre la función de respuesta V.
Pipeline (técnica de la clase):
    1. Filtro de media 3x3 (núcleo 1/9)
    2. Gradientes Sobel 3x3 (gx, gy)
    3. Componentes de la matriz de estructura: Ixx, Iyy, Ixy
    4. Suavizado gaussiano 3x3 de cada componente
    5. Respuesta de Harris: V = (A*B - C^2) - alpha*(A + B)^2
    6. Supresión de no máximos (dilatación 3x3) y umbral δ

δ es un umbral absoluto sobre V; no es una desviación estándar.

Requisitos:
pip install numpy opencv-python matplotlib
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CARPETA_EJERCICIO = Path.home() / "Desktop" / "ejercicio"
if not CARPETA_EJERCICIO.exists():
    CARPETA_EJERCICIO = Path.home() / "Escritorio" / "ejercicio"

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ALPHA_HARRIS = 0.04
RANGO_DELTA = (900, 1_000_000)
NUM_DELTAS_DEFECTO = 10
TAMANO_KERNEL_DILATACION = 5


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar_imagen_gris(ruta: Path) -> np.ndarray:
    """Carga en gris. np.fromfile evita problemas con acentos en Windows."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No fue posible decodificar: {ruta.name}")
    return img


def listar_imagenes(carpeta: Path) -> list[tuple[str, Path]]:
    """Devuelve (nombre_sin_extension, ruta) de cada imagen en la carpeta, ordenadas."""
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    archivos = sorted(
        p for p in carpeta.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN
    )
    if not archivos:
        raise FileNotFoundError(f"No hay imágenes en: {carpeta}")
    return [(p.stem, p) for p in archivos]


# ---------------------------------------------------------------------------
# Detector de Harris (técnica de la clase)
# ---------------------------------------------------------------------------

KERNEL_MEDIA_3X3 = np.ones((3, 3), dtype=np.float32) / 9.0

KERNEL_SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

KERNEL_SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
], dtype=np.float32)


def _anular_borde(matriz: np.ndarray) -> np.ndarray:
    """Pone a cero el borde de 1 píxel (in-place)."""
    matriz[0, :] = 0
    matriz[-1, :] = 0
    matriz[:, 0] = 0
    matriz[:, -1] = 0
    return matriz


def filtrar_media(img: np.ndarray) -> np.ndarray:
    """Paso 1. Media 3x3, redondeo a uint8, borde a cero."""
    suavizada = cv2.filter2D(
        img.astype(np.float32), -1, KERNEL_MEDIA_3X3,
        borderType=cv2.BORDER_CONSTANT,
    )
    return _anular_borde(np.round(suavizada).astype(np.uint8))


def gradientes_sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paso 2. Gradientes Sobel 3x3 en x e y, borde a cero."""
    entrada = img.astype(np.float32)
    gx = cv2.filter2D(entrada, -1, KERNEL_SOBEL_X, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.filter2D(entrada, -1, KERNEL_SOBEL_Y, borderType=cv2.BORDER_CONSTANT)
    return _anular_borde(gx), _anular_borde(gy)


def matriz_estructura(gx: np.ndarray, gy: np.ndarray):
    """Paso 3. Componentes Ixx, Iyy, Ixy."""
    return gx * gx, gy * gy, gx * gy


def suavizar_gaussiano(componente: np.ndarray) -> np.ndarray:
    """Paso 4. Gaussiano 3x3; sigma = 0 lo calcula OpenCV."""
    return cv2.GaussianBlur(componente, (3, 3), 0)


def respuesta_harris(img: np.ndarray, alpha: float = ALPHA_HARRIS) -> np.ndarray:
    """Pasos 1 a 5. Devuelve V = (A*B - C^2) - alpha*(A + B)^2."""
    suavizada = filtrar_media(img)
    gx, gy = gradientes_sobel(suavizada)
    ixx, iyy, ixy = matriz_estructura(gx, gy)

    A = suavizar_gaussiano(ixx)
    B = suavizar_gaussiano(iyy)
    C = suavizar_gaussiano(ixy)

    return (A * B - C ** 2) - alpha * (A + B) ** 2


def supresion_no_maximos(V: np.ndarray) -> np.ndarray:
    """Paso 6a. Dilatación 3x3 de V."""
    return cv2.dilate(V, np.ones((3, 3), np.uint8))


def detectar_esquinas(V: np.ndarray, V_max: np.ndarray, delta: float):
    """
    Paso 6b. Umbral absoluto δ.
    Devuelve (puntual, dilatada, num_esquinas).
    """
    puntual = ((V == V_max) & (V >= delta)).astype(np.uint8)
    kernel = np.ones((TAMANO_KERNEL_DILATACION, TAMANO_KERNEL_DILATACION), np.uint8)
    dilatada = cv2.dilate(puntual, kernel)
    return puntual, dilatada, int(puntual.sum())


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

COLOR_MARCA_RGB = (255, 30, 30)


def superponer_mascara(img_gris: np.ndarray, mascara: np.ndarray, grosor: int = 1) -> np.ndarray:
    """Pinta la máscara en rojo sobre la imagen en gris (RGB)."""
    salida = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2RGB)
    if grosor > 1:
        mascara = cv2.dilate(mascara, np.ones((grosor, grosor), np.uint8))
    salida[mascara > 0] = COLOR_MARCA_RGB
    return salida


def generar_vista(img_gris: np.ndarray, mascara: np.ndarray, modo: str, tipo: str) -> np.ndarray:
    """modo: 'overlay' o 'binary'. tipo: 'puntual' o 'dilatada'."""
    if modo == "binary":
        return mascara * 255
    grosor = 2 if tipo == "puntual" else 1
    return superponer_mascara(img_gris, mascara, grosor)


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

@dataclass
class ResultadoImagen:
    nombre: str
    shape: tuple
    conteos: list
    vistas: dict


def generar_deltas(num: int, escala: str) -> np.ndarray:
    """Valores de δ en RANGO_DELTA, escala 'log' o 'linear'."""
    inicio, fin = RANGO_DELTA
    if escala == "log":
        return np.geomspace(inicio, fin, num).astype(int)
    return np.linspace(inicio, fin, num).astype(int)


def procesar_imagen(nombre: str, ruta: Path, deltas: np.ndarray) -> ResultadoImagen:
    img = cargar_imagen_gris(ruta)
    V = respuesta_harris(img)
    V_max = supresion_no_maximos(V)

    vistas = {(t, m): [] for t in ("puntual", "dilatada") for m in ("overlay", "binary")}
    conteos = []

    for delta in deltas:
        puntual, dilatada, n = detectar_esquinas(V, V_max, delta)
        conteos.append(n)
        for tipo, mascara in (("puntual", puntual), ("dilatada", dilatada)):
            for modo in ("overlay", "binary"):
                vistas[(tipo, modo)].append(generar_vista(img, mascara, modo, tipo))

    return ResultadoImagen(nombre, img.shape, conteos, vistas)


def formatear_delta(delta: int) -> str:
    return f"{delta:,}".replace(",", " ")


def tabla_markdown(resultados: list[ResultadoImagen], deltas: np.ndarray) -> str:
    encabezado = "| Imagen | " + " | ".join(f"δ={formatear_delta(d)}" for d in deltas) + " |"
    separador = "| :--- | " + " | ".join([":---:"] * len(deltas)) + " |"
    filas = [
        f"| **{r.nombre}** | " + " | ".join(f"{c:,}" for c in r.conteos) + " |"
        for r in resultados
    ]
    return "\n".join([encabezado, separador, *filas])


def graficar_tabla(resultados, deltas, tipo, modo, ruta_salida: Path) -> None:
    filas, columnas = len(resultados), len(deltas)
    fig, ejes = plt.subplots(filas, columnas, figsize=(26, 16), dpi=150, squeeze=False)

    etiqueta_tipo = "Esquinas dilatadas (5x5)" if tipo == "dilatada" else "Esquinas puntuales"
    etiqueta_modo = "superpuestas en la imagen" if modo == "overlay" else "máscara binaria"
    fig.suptitle(
        "Detector de Harris: barrido del umbral δ sobre la respuesta V\n"
        f"{etiqueta_tipo}, {etiqueta_modo}",
        fontsize=15, fontweight="bold",
    )

    for i, res in enumerate(resultados):
        for j, delta in enumerate(deltas):
            ax = ejes[i, j]
            vista = res.vistas[(tipo, modo)][j]
            if modo == "binary":
                ax.imshow(vista, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(vista)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(f"δ = {formatear_delta(delta)}", fontsize=9.5, fontweight="bold")
            if j == 0:
                alto, ancho = res.shape
                ax.set_ylabel(f"{res.nombre}\n({ancho}x{alto})", fontsize=9.5,
                              fontweight="bold", rotation=0, labelpad=45, va="center")

            ax.text(0.96, 0.05, f"N={res.conteos[j]:,}", transform=ax.transAxes,
                    fontsize=8, color="white", fontweight="bold", ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                              alpha=0.7, edgecolor="none"))

    plt.tight_layout(rect=[0.05, 0.02, 0.99, 0.95])
    fig.savefig(ruta_salida, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura guardada: {ruta_salida}")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Barrido del umbral δ del detector de Harris."
    )
    parser.add_argument("--num-deltas", type=int, default=NUM_DELTAS_DEFECTO)
    parser.add_argument("--escala", choices=["log", "linear"], default="log")
    parser.add_argument("--carpeta", type=Path, default=CARPETA_EJERCICIO,
                        help="Carpeta con las imágenes (por defecto: ejercicio en el escritorio).")
    parser.add_argument("--salida", type=Path, default=None,
                        help="Carpeta de salida (por defecto: resultados_grid dentro de la carpeta).")
    parser.add_argument("--mostrar", action="store_true",
                        help="Muestra las figuras al terminar.")
    return parser


def main() -> None:
    args = construir_parser().parse_args()

    carpeta = args.carpeta
    salida = args.salida or (carpeta / "resultados_grid")
    salida.mkdir(parents=True, exist_ok=True)

    deltas = generar_deltas(args.num_deltas, args.escala)
    imagenes = listar_imagenes(carpeta)

    print(f"[*] Carpeta: {carpeta}")
    print(f"[*] Imágenes ({len(imagenes)}): {[n for n, _ in imagenes]}")
    print(f"[*] δ ({args.escala}, N={args.num_deltas}): {list(map(int, deltas))}")

    resultados = []
    for nombre, ruta in imagenes:
        print(f"[+] Procesando {nombre} ({ruta.name})")
        resultados.append(procesar_imagen(nombre, ruta, deltas))

    md = tabla_markdown(resultados, deltas)
    print("\n" + md + "\n")

    ruta_md = salida / f"tabla_conteo_{args.num_deltas}_deltas_{args.escala}.md"
    ruta_md.write_text(
        f"# Conteo de esquinas por umbral δ ({args.num_deltas} valores, escala {args.escala})\n\n{md}\n",
        encoding="utf-8",
    )
    print(f"[OK] Tabla guardada: {ruta_md}")

    for tipo in ("puntual", "dilatada"):
        for modo in ("overlay", "binary"):
            graficar_tabla(resultados, deltas, tipo, modo,
                           salida / f"plot_{tipo}_{modo}_{args.escala}.png")

    if args.mostrar:
        plt.show()

    print("\n[PROCESO COMPLETADO]")


if __name__ == "__main__":
    main()