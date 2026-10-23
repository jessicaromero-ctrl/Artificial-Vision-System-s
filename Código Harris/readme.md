# Experimento: detección de esquinas con el detector de Harris

Este experimento aplica el detector de esquinas de Harris a seis imágenes en escala de grises y varía el umbral absoluto **δ** sobre la función de respuesta **V**. Para cada combinación (imagen, δ) se cuenta el número de esquinas detectadas y se generan visualizaciones.

## 1. Conceptos

### 1.1 Esquina de Harris
Una esquina es un punto donde el brillo cambia de forma significativa en dos direcciones. El detector de Harris mide esto a partir de la **matriz de estructura** (tensor de gradiente), construida con las derivadas parciales de la imagen en cada píxel.

### 1.2 Matriz de estructura
A partir de los gradientes horizontal (`gx`) y vertical (`gy`):

- `Ixx = gx²`
- `Iyy = gy²`
- `Ixy = gx · gy`

Tras el suavizado gaussiano, estas componentes forman la matriz simétrica

```
M = | A  C |
    | C  B |
```

con `A = Ixx`, `B = Iyy` y `C = Ixy`.

### 1.3 Función de respuesta de Harris

```
V = (A·B − C²) − α · (A + B)²
```

`A·B − C²` es el determinante de `M`, `(A + B)` es su traza y `α = 0.04` es una constante empírica. Valores grandes de `V` indican esquinas; valores negativos indican bordes; valores pequeños indican regiones planas.

### 1.4 Umbral δ
**δ no es una desviación estándar ni un parámetro del suavizado.** δ es un **umbral absoluto** sobre `V`: solo se conservan los máximos locales con `V ≥ δ`. Subir δ reduce el número de esquinas; bajarlo lo aumenta.

El rango `[900, 1 000 000]` viene del código de referencia de la clase. Debe verificarse con los conteos reales antes de usarlo como conclusión.

### 1.5 Esquinas puntuales y dilatadas
- **Puntuales:** máximos locales de `V` (vecindad 3×3) que superan δ.
- **Dilatadas:** las puntuales ensanchadas con un kernel 5×5, para verlas al reducir la resolución de la figura.

### 1.6 Superposición y máscara binaria
- **Superposición:** esquinas en rojo sobre la imagen original.
- **Binaria:** máscara blanca sobre fondo negro.

## 2. Pipeline de cálculo

| Paso | Operación | Detalle |
| :--- | :--- | :--- |
| 1 | Filtro de media | Núcleo 3×3 de valores 1/9, bordes a cero, redondeo a `uint8` |
| 2 | Gradientes Sobel | Núcleos 3×3 `gx` y `gy`, bordes a cero |
| 3 | Matriz de estructura | `Ixx = gx²`, `Iyy = gy²`, `Ixy = gx·gy` |
| 4 | Suavizado gaussiano | Núcleo 3×3 con `sigma = 0` (lo calcula OpenCV) sobre cada componente |
| 5 | Respuesta de Harris | `V = (A·B − C²) − α·(A + B)²` con `α = 0.04` |
| 6 | Supresión de no máximos | Dilatación 3×3 de `V`; se conserva `V == V_max` |
| 7 | Umbral δ | Se conservan los píxeles con `V ≥ δ` |
| 8 | Dilatación para visualización | Kernel 5×5 sobre la máscara puntual |

## 3. Imágenes

| Nombre | Archivo |
| :--- | :--- |
| Cuadrado | `Cuadrado.jpeg` |
| Cuadrícula | `Cuadrícula.jpeg` |
| Estrella | `Estrella.jpeg` |
| Figuras | `Figuras.jpeg` |
| Lena_Gray | `Lena_Gray.png` |
| Libros | `Libros.jpeg` |

Las imágenes deben estar en la misma carpeta que `harris_experimento.py`. Con `--incluir-extra` se añaden, si existen, cuatro imágenes de la carpeta superior.

## 4. Requisitos

```bash
pip install numpy opencv-python matplotlib
```

## 5. Ejecución

Por defecto (10 valores de δ en escala logarítmica, seis imágenes):

```bash
python harris_experimento.py
```

| Opción | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `--num-deltas N` | Número de valores de δ | `10` |
| `--escala {log,linear}` | Distribución de δ en el rango | `log` |
| `--incluir-extra` | Incluye imágenes adicionales si existen | desactivado |
| `--salida DIR` | Carpeta de salida | `resultados_grid/` |
| `--mostrar` | Abre las figuras en ventana interactiva | desactivado |

Ejemplo con escala lineal y 6 valores:

```bash
python harris_experimento.py --num-deltas 6 --escala linear
```

## 6. Salidas

Se guardan en `resultados_grid/` (o en la carpeta de `--salida`):

| Archivo | Contenido |
| :--- | :--- |
| `tabla_conteo_<N>_deltas_<escala>.md` | Conteo de esquinas por imagen y δ |
| `plot_puntual_overlay_<escala>.png` | Esquinas puntuales superpuestas |
| `plot_dilatada_overlay_<escala>.png` | Esquinas dilatadas superpuestas |
| `plot_puntual_binary_<escala>.png` | Máscara binaria de esquinas puntuales |
| `plot_dilatada_binary_<escala>.png` | Máscara binaria de esquinas dilatadas |

## 7. Pasos de optimización y lectura de resultados

1. **Verificar la carga.** Si una imagen no existe, el script se detiene con un mensaje claro. Revisar que los nombres coincidan exactamente, incluidos los acentos.
2. **Calibrar δ.** Si el conteo es cero para todos los valores, δ está por encima del máximo de `V`. Si el conteo es muy alto, δ está demasiado bajo. Explorar con `--escala log` y refinar con `--escala linear` en el intervalo de interés.
3. **Controlar el ruido.** Un conteo que crece de forma abrupta al bajar δ suele indicar respuesta a texturas y ruido. Comparar con la estrella y la cuadrícula, que tienen esquinas bien definidas.
4. **Interpretar la dilatación.** Las vistas dilatadas muestran zonas, no puntos. Para contar, usar siempre la vista puntual.
5. **Reducir el costo.** `V` se calcula una vez por imagen; cambiar δ solo repite el umbral y la dilatación. Si el tiempo es un problema, reducir `--num-deltas` o procesar menos imágenes.
6. **Documentar δ como umbral.** En cualquier reporte, describir δ como umbral absoluto de `V`, no como parámetro de suavizado.
