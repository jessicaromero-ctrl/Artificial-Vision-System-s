# Detección de baches mediante Procesamiento Digital de Imágenes (PDI)

Pipeline de PDI clásico para el análisis preliminar de imágenes de pavimento con baches. Constituye la **etapa determinista** de una arquitectura híbrida: reduce la región de análisis y extrae características geométricas antes de cualquier modelo de Machine Learning o Deep Learning.

---

## 1. Contexto y problema

En México, el deterioro del pavimento incrementa los accidentes, los daños a vehículos y las dificultades de movilidad para peatones, automovilistas y transporte público. El diagnóstico manual de la infraestructura es lento y costoso; un sistema automatizado permite priorizar intervenciones con base en evidencia visual.

## 2. Objetivos

**General.** Diseñar y evaluar un pipeline de PDI que caracterice imágenes de baches a partir de bordes, esquinas y binarización.

**Específicos.**

1. Convertir las imágenes de RGB (BGR en OpenCV) a escala de grises.
2. Estandarizar las dimensiones de entrada.
3. Aplicar técnicas de suavizado para reducir ruido.
4. Detectar bordes con los operadores Sobel, Prewitt y Roberts, y compararlos cuantitativamente.
5. Detectar esquinas y puntos de interés con Harris y Shi-Tomasi.
6. Binarizar la imagen y evaluar la segmentación frente a las anotaciones.

## 3. Dataset

- **Nombre:** Pothole Detection (Kaggle, `andrewmvd/pothole-detection`)
- **Enlace:** https://www.kaggle.com/datasets/andrewmvd/pothole-detection
- **Contenido:** 665 imágenes con anotaciones en formato PASCAL VOC (XML), con cajas delimitadoras de cada bache.

El dataset no se incluye en el repositorio por su tamaño (≈334 MB).

## 4. Estructura del proyecto

```
pothole-detection-pipeline/
├── pipeline/
│   └── pothole_pipeline.py    # Pipeline completo
├── download_dataset.py        # Descarga del dataset con kagglehub
├── requirements.txt           # Dependencias
├── .gitignore
├── README.md
└── resultados/                # Generado al ejecutar (no versionado)
    ├── metricas_por_imagen.csv
    ├── resumen.csv
    └── paneles/               # Visualizaciones por etapa
```

## 5. Requisitos e instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencias: `opencv-python`, `numpy`, `kagglehub`.

## 6. Ejecución

**Descarga del dataset** (requiere credenciales de Kaggle en la variable `KAGGLE_API_TOKEN` o en `~/.kaggle/kaggle.json`):

```bash
python download_dataset.py
```

**Prueba con una muestra:**

```bash
python pipeline/pothole_pipeline.py --images "RUTA/images" --annotations "RUTA/annotations" --out resultados --limit 20
```

**Ejecución completa:**

```bash
python pipeline/pothole_pipeline.py --images "RUTA/images" --annotations "RUTA/annotations" --out resultados --viz 5
```

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `--images` | Carpeta con las imágenes | obligatorio |
| `--annotations` | Carpeta con XML VOC (evaluación) | ninguno |
| `--out` | Carpeta de salida | `resultados_pothole` |
| `--size` | Lado de la imagen estandarizada (px) | 512 |
| `--smooth` | `gaussian`, `median` o `bilateral` | `gaussian` |
| `--limit` | Procesar solo N imágenes (0 = todas) | 0 |
| `--viz` | Número de paneles de visualización | 5 |

## 7. Pipeline paso a paso

Sea $I$ la imagen de entrada con dimensiones $W \times H$.

### Etapa 1. Espacio de color

Conversión de BGR a escala de grises mediante la luminancia ponderada:

$$Y(x,y) = 0.299\,R(x,y) + 0.587\,G(x,y) + 0.114\,B(x,y)$$

Se reduce la dimensionalidad de 3 canales a 1 y se elimina la dependencia del color, que en pavimento aporta poca información discriminante.

### Etapa 2. Estandarización de dimensiones (letterbox)

Escala uniforme $s = \min\left(\frac{S}{W}, \frac{S}{H}\right)$ con $S = 512$, seguida de relleno gris (valor 114) hasta un lienzo cuadrado de $S \times S$. Las coordenadas de las cajas se transforman con el mismo factor $s$ y el mismo desplazamiento de relleno. A diferencia de un redimensionado directo, esta transformación conserva la relación de aspecto del bache.

### Etapa 3. Suavizado

Convolución con un núcleo gaussiano de $5 \times 5$ y $\sigma = 1.0$:

$$G(x,y) = \frac{1}{2\pi\sigma^2}\, e^{-\frac{x^2+y^2}{2\sigma^2}}$$

$$I_s(x,y) = \sum_{i=-2}^{2}\sum_{j=-2}^{2} I(x+i,\, y+j)\cdot K(i,j)$$

El suavizado atenúa el ruido de alta frecuencia (textura del asfalto) que de otro modo generaría bordes falsos.

### Etapa 4. Detección de bordes

Cada operador estima el gradiente con dos núcleos (horizontal y vertical). La magnitud se calcula como:

$$M(x,y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$

| Operador | $G_x$ | $G_y$ | Tamaño |
|---|---|---|---|
| Sobel | $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$ | $\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$ | 3×3 |
| Prewitt | $\begin{bmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{bmatrix}$ | $\begin{bmatrix}-1&-1&-1\\0&0&0\\1&1&1\end{bmatrix}$ | 3×3 |
| Roberts | $\begin{bmatrix}1&0\\0&-1\end{bmatrix}$ | $\begin{bmatrix}0&1\\-1&0\end{bmatrix}$ | 2×2 |

Sobel pondera más el píxel central y Roberts usa solo cuatro píxeles, por lo que es el más sensible al ruido. La magnitud se normaliza a $[0, 255]$ y se binariza con el umbral de Otsu para obtener el mapa de bordes.

### Etapa 5. Esquinas y puntos de interés

- **Harris.** Respuesta por píxel $R = \det(M) - k\,\mathrm{tr}(M)^2$, con $k = 0.04$, donde $M$ es la matriz de autocorrelación del gradiente. Se conservan los máximos locales con $R > 0.01 \cdot \max(R)$.
- **Shi-Tomasi.** Criterio $\lambda_{\min} > q \cdot \max(\lambda_{\min})$ con $q = 0.01$, donde $\lambda_{\min}$ es el menor autovalor de $M$. Se limita a 150 puntos con distancia mínima de 7 px.

### Etapa 6. Binarización

Se generan dos máscaras, con $255$ para regiones oscuras (candidatas a bache):

1. **Otsu global.** Elige el umbral $t^*$ que maximiza la varianza entre clases: $\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,[\mu_0(t) - \mu_1(t)]^2$.
2. **Umbral adaptativo gaussiano.** Ventana de 31 px y constante $C = 10$, para compensar la iluminación no uniforme.

Ambas máscaras se limpian con apertura y cierre morfológicos (elemento estructural elíptico de $3 \times 3$).

### Evaluación

Si existen anotaciones, las cajas se convierten en una máscara de referencia $A$ y se calculan:

$$\mathrm{IoU}(P, A) = \frac{|P \cap A|}{|P \cup A|}$$

- **Energía en ROI:** fracción de la magnitud del gradiente que cae dentro de la caja.
- **Precisión en ROI:** fracción de píxeles de borde que caen dentro de la caja.
- **Cajas con punto:** fracción de cajas con al menos un punto de interés.

## 8. Glosario

| Término | Definición |
|---|---|
| **BGR / RGB** | Orden de canales de color. OpenCV usa BGR. |
| **Escala de grises** | Imagen de un canal que representa la luminancia. |
| **Letterbox** | Redimensionado que conserva la relación de aspecto y rellena los bordes. |
| **Convolución** | Operación que combina una vecindad de píxeles con un núcleo (kernel) para producir una nueva imagen. |
| **Núcleo (kernel)** | Matriz pequeña de pesos que define la operación de convolución. |
| **Suavizado gaussiano** | Convolución con un núcleo gaussiano; atenúa el ruido de alta frecuencia. |
| **Gradiente** | Vector de derivadas parciales que indica la dirección y la intensidad del cambio de intensidad. |
| **Magnitud del gradiente** | Norma del gradiente; mide qué tan abrupto es el cambio en un píxel. |
| **Borde** | Conjunto de píxeles donde la magnitud del gradiente es alta. |
| **Sobel / Prewitt / Roberts** | Operadores de gradiente de 3×3, 3×3 y 2×2 respectivamente. |
| **Umbral (threshold)** | Valor que separa píxeles en dos clases según su intensidad. |
| **Umbral de Otsu** | Umbral que maximiza la varianza entre las dos clases de píxeles. |
| **Umbral adaptativo** | Umbral que se calcula localmente, por ventana, para tolerar iluminación desigual. |
| **Binarización** | Conversión de la imagen a dos valores (0 y 255). |
| **Morfología matemática** | Operaciones sobre formas binarias. *Apertura* elimina puntos aislados; *cierre* rellena huecos pequeños. |
| **Esquina** | Punto donde el gradiente cambia en dos direcciones; más estable que un borde para localizar. |
| **Harris** | Detector de esquinas basado en la matriz de autocorrelación del gradiente. |
| **Shi-Tomasi** | Variante de Harris que usa el menor autovalor de la autocorrelación como criterio. |
| **Autovalor** | Valor escalar asociado a una matriz; en este contexto mide la intensidad del gradiente en la dirección más débil. |
| **ROI (Region of Interest)** | Región de la imagen que se analiza con prioridad; aquí, la caja del bache. |
| **IoU (Intersection over Union)** | Métrica de solapamiento entre dos regiones: intersección dividida entre unión. |
| **PASCAL VOC (XML)** | Formato estándar de anotaciones con cajas delimitadoras. |
| **Densidad de bordes** | Proporción de píxeles que son borde en la imagen. |
| **Línea base (baseline)** | Resultado de referencia contra el que se compara una métrica; por ejemplo, el área que ocupan las cajas. |

## 9. Resultados (665 imágenes, 512×512, suavizado gaussiano)

Las 665 imágenes se procesaron sin omisiones.

### Comparación de operadores de borde

| Operador | Tiempo (ms) | Densidad de bordes | Energía en ROI | Precisión en ROI |
|---|---|---|---|---|
| Sobel | 0.914 | 0.1740 | 0.2080 | 0.2437 |
| Prewitt | 0.895 | 0.1734 | 0.2080 | 0.2438 |
| Roberts | 0.785 | 0.1763 | 0.2073 | 0.2422 |

### Esquinas

| Detector | Puntos (media) | Puntos en ROI | Cajas con al menos un punto |
|---|---|---|---|
| Shi-Tomasi | 150.0 (límite) | 0.332 | 0.938 |
| Harris | 841.6 | 0.300 | 0.981 |

### Binarización

| Método | IoU medio con la máscara de bache | Fracción de píxeles en primer plano |
|---|---|---|
| Otsu global | 0.1193 | 0.530 |
| Adaptativo | 0.1080 | 0.096 |

## 10. Interpretación y limitaciones

1. **Los tres operadores son estadísticamente indistinguibles en estas métricas.** Las diferencias en energía en ROI y precisión en ROI son inferiores a 0.003. Sobel y Prewitt producen mapas casi idénticos, y Roberts es el más rápido (≈15 % menos tiempo). Para afirmar diferencias habría que aplicar una prueba estadística sobre las métricas por imagen.

2. **Falta una línea base.** La energía en ROI de 0.21 solo es significativa si se compara con la fracción del área total que ocupan las cajas. Si las cajas cubren, por ejemplo, 20 % de la imagen, un operador aleatorio obtendría un valor cercano a 0.20. Es una limitación del diseño de evaluación actual y debe corregirse antes de concluir cuál operador es mejor.

3. **La binarización no segmenta los baches de forma útil.** Un IoU de 0.11–0.12 indica que la segmentación no coincide con las anotaciones. Otsu marca como primer plano el 53 % de la imagen, es decir, el umbral es demasiado permisivo para este tipo de escena.

4. **Shi-Tomasi está saturado.** Siempre devuelve el límite de 150 puntos, por lo que su conteo no aporta información. Su proporción en ROI (0.33) debe interpretarse con esa restricción.

5. **Harris tiene una cobertura engañosa.** Detecta casi 842 puntos por imagen; la alta proporción de cajas con punto (0.98) es consecuencia del volumen, no de la calidad. La proporción de puntos dentro de la ROI (0.30) es la métrica relevante, y es baja.

**Conclusión preliminar.** El PDI clásico con estos parámetros no basta para localizar baches. Sirve como generador de regiones candidatas y como base de comparación para el modelo híbrido, pero requiere mejoras antes de integrarse a una etapa de clasificación.

## 11. Trabajo futuro

- Añadir una línea base aleatoria y el área de las cajas como referencia en todas las métricas.
- Corregir la iluminación con CLAHE antes de la binarización.
- Restringir el análisis a una región de carretera para reducir falsos positivos fuera del pavimento.
- Filtrar las regiones candidatas por área geométrica y relación de aspecto.
- Pasar los recortes de las regiones candidatas a un modelo ligero (por ejemplo, YOLOv8-nano o MobileNet) para clasificación.
- Ejecutar pruebas estadísticas (por ejemplo, Wilcoxon de muestras pareadas) sobre las métricas por imagen para comparar los operadores.

## 12. Referencias

- Sobel, I. & Feldman, G. (1968). *A 3×3 Isotropic Gradient Operator for Image Processing*. Presentado en Stanford Artificial Intelligence Project.
- Prewitt, J. M. S. (1970). Object enhancement and extraction. En *Picture Processing and Psychopictorics*. Academic Press.
- Roberts, L. G. (1963). *Machine Perception of Three-Dimensional Solids*. MIT Lincoln Laboratory.
- Harris, C. & Stephens, M. (1988). A combined corner and edge detector. *Proceedings of the 4th Alvey Vision Conference*.
- Shi, J. & Tomasi, C. (1994). Good Features to Track. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
- Documentación de OpenCV: https://docs.opencv.org/
