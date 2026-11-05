# Sistemas de Visión Artificial: Prácticas

Repositorio de prácticas de la asignatura **Sistemas de Visión Artificial**, Maestría en Inteligencia Artificial, Universidad Politécnica Metropolitana del Estado de Hidalgo (UPMH), periodo 2026.

| Dato | Detalle |
|---|---|
| Programa | Maestría en Inteligencia Artificial |
| Institución | Universidad Politécnica Metropolitana del Estado de Hidalgo (UPMH) |
| Asignatura | Sistemas de Visión Artificial |
| Profesor | Dr. Ernesto García Amaro |
| Alumna | Jessica Melani Romero Lora |
| Periodo | 2026 |

> ⚠️ **Aviso de uso académico y de datos**
>
> - Este repositorio contiene prácticas de la Maestría en Inteligencia Artificial de la UPMH. Su contenido tiene fines exclusivamente académicos y educativos.
> - La práctica 2 usa fotografías clínicas de lesiones cutáneas de pacientes con neurofibromatosis tipo 1 (NF1). Estas imágenes **no se incluyen** en este repositorio. No deben redistribuirse ni publicarse sin autorización de la institución y de las personas involucradas.
> - Los documentos y el código están protegidos como material académico. Se prohíbe su reproducción o distribución total o parcial sin autorización de la autora.

---

## Objetivos

- Documentar, de forma reproducible, las prácticas de procesamiento digital de imágenes realizadas en la asignatura.
- Comparar técnicas clásicas de PDI con métricas cuantitativas, en lugar de evaluarlas solo de forma visual.
- Servir como base para el proyecto final, que incorporará modelos de aprendizaje profundo a un pipeline híbrido.

## Contenido del repositorio

## Prácticas

| # | Práctica | Carpeta | Documento | Código | Estado |
|---|---|---|---|---|---|
| 1 | Análisis comparativo de filtros paso bajo espaciales | [`01-filtros-paso-bajo/`](01-filtros-paso-bajo/) | `Reporte_Filtros_Paso_Bajo.pdf` | Pendiente de incluir | Reporte disponible |
| 2 | Operaciones aritméticas y lógicas a nivel de píxel | [`02-operaciones-pixeles/`](02-operaciones-pixeles/) | `Reporte_Operaciones_Pixeles.pdf` | `practica_operaciones_pixeles.py` | Reporte y código disponibles |
| 3 | Detección de baches mediante PDI | [`03-deteccion-baches/`](03-deteccion-baches/) | Documento en revisión | `pipeline/pothole_pipeline.py` | Pipeline, métricas y figuras disponibles |
| — | Ejercicio: detección de esquinas (Harris) | [`Código Harris/`](Código%20Harris/) | Sin documento | Ejercicio | En preparación |

### Práctica 1. Análisis comparativo de filtros paso bajo espaciales

Se comparan tres filtros de promedio y suavizado aplicados por convolución discreta:

- **Media 1/9:** núcleo 3×3 con pesos uniformes.
- **Gaussiano 1/16:** núcleo 3×3 con pesos binomiales, `[1 2 1; 2 4 2; 1 2 1]`.
- **Gaussiano 1/273:** núcleo 5×5 con pesos binomiales, normalizado por su suma (273).

**Análisis:**

- Respuesta en frecuencia de cada filtro, en una y en dos dimensiones, con su ganancia en decibeles.
- Efecto sobre cuatro imágenes de prueba: Lena (color), Lena (escala de grises), un paisaje y un paisaje con ruido.
- Métricas de calidad: MSE y PSNR respecto a la imagen original, SSIM, varianza del Laplaciano (como medida de nitidez) y magnitud del gradiente de Sobel (como medida de bordes).

**Resultado principal:** el Gaussiano 1/273 produce el mayor suavizado, a costa de perder más detalle en las frecuencias altas.

### Práctica 2. Operaciones aritméticas y lógicas a nivel de píxel

Se aplican las operaciones SUMA, RESTA, AND y OR entre imágenes sobre un conjunto de **7 imágenes dermatológicas**, preprocesadas a 512 × 512 píxeles. Las imágenes se clasifican en máculas (img1, img2, img4 y img6) y nódulos (img3, img5 y img7). La fuente de las imágenes es DermNet / Te Whatu Ora Health New Zealand (página *Neurofibromatosis images*).

El preprocesamiento descarta el canal alfa, elimina la franja de marca de agua de la izquierda en las imágenes que la tienen, recorta un cuadrado centrado y reescala con interpolación por área.

**Pares de trabajo**, elegidos por su contenido y no al azar:

| Par | Imágenes | Propósito |
|---|---|---|
| P1 | img3, img5 | Misma escena a distinta escala: resta con registro previo |
| P2 | img4, img6 | Dos máculas claras: suma saturada frente a suma modular |
| P3 | img1, img7 | Clases distintas con rangos dinámicos muy diferentes |

**Operaciones implementadas:**

- **Suma:** saturada, modular (módulo 256), ponderada con α = 0.5, suma de un escalar como ajuste de brillo y barrido de α en el par P3.
- **Resta:** registro afín con ECC (`cv2.findTransformECC`) antes de restar en el par P1; resta direccional y valor absoluto en P2; sustracción de fondo con la diferencia entre dos filtros gaussianos (σ = 3 y σ = 35) sobre la luminancia invertida.
- **AND:** entre imágenes en los pares P2 y P3; planos de bits con máscaras de 1 a 8 bits; máscara de lesión mediante Otsu sobre el residuo corregido por fondo, con apertura y cierre morfológicos; extracción de la región de interés; IoU entre máscaras.
- **OR:** entre imágenes en los pares P2 y P3; unión de máscaras y fotomontaje de dos lesiones.
- **Combinación AND + OR:** ocultamiento de los cuatro bits altos de una imagen en los cuatro bits bajos de otra, con verificación de la recuperación.

**Verificación numérica.** El script comprueba sobre los datos reales las identidades siguientes y detiene la ejecución si alguna falla: *a + b = (a OR b) + (a AND b)*, *a AND b ≤ min(a, b)*, *a OR b ≥ max(a, b)*, igualdad entre suma saturada y modular cuando no hay desbordamiento, *|a − b| = (a − b)⁺ + (b − a)⁺* y elementos neutros.

**Requisitos:** Python 3.9 o superior, NumPy, OpenCV (`opencv-python`) y Matplotlib. Están listados en `02-operaciones-pixeles/requirements.txt`.

**Ejecución:**

```bash
cd 02-operaciones-pixeles
python practica_operaciones_pixeles.py --entrada RUTA_A_LAS_IMAGENES --salida resultados --tam 512
```

La carpeta de entrada debe contener los archivos `img1.png` a `img7.png`.

**Salidas** (dentro de `resultados/`):

- `figuras/`: 16 figuras compuestas, de `fig01` a `fig16`.
- `imagenes/`: cada resultado por separado, en PNG sin pérdida.
- `metricas.csv`: todas las métricas en formato largo (operación, contexto, métrica, valor).

**Datos:** las imágenes clínicas no se versionan. Colócalas en una carpeta local ignorada por Git, como se indica en el aviso del inicio.

## Práctica 3. Detección de baches mediante PDI

Pipeline de PDI clásico para identificar regiones candidatas a bache en imágenes de pavimento. Es la etapa determinista de una arquitectura híbrida: reduce la región de análisis antes de cualquier modelo de aprendizaje profundo.

**Dataset:** Pothole Detection (Kaggle, `andrewmvd/pothole-detection`), 665 imágenes con anotaciones en formato PASCAL VOC.

**Etapas del pipeline:**

1. Conversión a escala de grises.
2. Estandarización a 512 × 512 píxeles con letterbox.
3. Suavizado gaussiano.
4. Detección de bordes con Sobel, Prewitt y Roberts.
5. Detección de esquinas con Harris y Shi-Tomasi.
6. Binarización con umbral de Otsu y umbral adaptativo.

**Resultados principales (665 imágenes):**

| Indicador | Resultado | Lectura |
|---|---|---|
| Operadores de borde | Sobel, Prewitt y Roberts equivalentes | Roberts es el más rápido (≈0.79 ms frente a ≈0.91 ms de Sobel) |
| Esquinas | Shi-Tomasi limitado a 150 puntos; Harris produce ≈842 puntos por imagen | Ninguno localiza baches de forma directa |
| Binarización | IoU medio de 0.119 (Otsu) y 0.108 (adaptativo) | Insuficiente como detector final |

**Conclusión:** el PDI clásico, con estos parámetros, no basta para localizar baches. Sirve como generador de regiones candidatas y como línea base para el modelo híbrido del proyecto final. La evaluación carece aún de una línea base de área, por lo que las diferencias entre operadores no son concluyentes.

**Documentación completa:** [`03-deteccion-baches/README.md`](03-deteccion-baches/README.md), con instalación, ejecución, descripción de cada etapa y glosario.

**Figuras:** [`03-deteccion-baches/resultados/graficos/`](03-deteccion-baches/resultados/graficos/).

## Requisitos generales

- Python 3.9 o superior.
- Cada práctica indica sus dependencias en su propio README o en `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt  # en la carpeta de la práctica correspondiente
```

## Estado del repositorio

- [x] Reporte PDF de la práctica 1.
- [x] Reporte PDF de la práctica 2.
- [x] Código de la práctica 2.
- [x] Pipeline, métricas y figuras de la práctica 3.
- [ ] Código de la práctica 1.
- [ ] Documento Word y PDF final de la práctica 3, pendientes de revisión y aprobación.
- [ ] Documentación del ejercicio de Harris.

## Autoría

Jessica Melani Romero Lora, Maestría en Inteligencia Artificial, UPMH, 2026.

Este material tiene fines exclusivamente académicos. Para cualquier otro uso, consultar el aviso del inicio.
