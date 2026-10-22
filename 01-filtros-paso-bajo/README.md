# Análisis comparativo de filtros paso bajo espaciales

| | |
|---|---|
| **Asignatura** | Sistemas de Visión Artificial |
| **Programa** | Maestría en Inteligencia Artificial |
| **Institución** | Universidad Politécnica Metropolitana del Estado de Hidalgo (UPMH) |
| **Profesor** | Dr. Ernesto García Amaro |
| **Alumna** | Jessica Melani Romero Lora |
| **Periodo** | 2026 |
| **Actividad** | 1. Filtros paso bajo espaciales |

---

> ⚠️ **AVISO DE USO ACADÉMICO**
>
> - Esta práctica forma parte de la **Maestría en Inteligencia Artificial de la UPMH** y tiene **fines exclusivamente académicos y educativos**.
> - El contenido está **protegido como material académico**. Se prohíbe su reproducción o distribución total o parcial sin autorización de la autora.

---

## 1. Descripción

La práctica compara tres filtros de promediado y suavizado aplicados por convolución discreta sobre imágenes en color y en escala de grises. El objetivo es observar cómo cada kernel atenúa las frecuencias altas, que corresponden a bordes, detalles y ruido.

## 2. Filtros evaluados

| Filtro | Kernel | Normalización |
|---|---|---|
| Media | 3×3, pesos uniformes | 1/9 |
| Gaussiano 3×3 | `[1 2 1; 2 4 2; 1 2 1]` | 1/16 |
| Gaussiano 5×5 | Pesos binomiales de 5×5 | 1/273 |

Cada kernel se normaliza para que la suma de sus pesos sea 1 y así no altere el brillo medio de la imagen.

## 3. Análisis realizado

- **Respuesta en frecuencia:** de cada filtro, en una dimensión y en dos dimensiones, con su ganancia expresada en dB.
- **Efecto sobre imágenes de prueba:** Lena (RGB), Lena en escala de grises, paisaje y paisaje con ruido.
- **Métricas de calidad:**
  - **MSE y PSNR:** diferencia respecto a la imagen original.
  - **SSIM:** similitud estructural con la original.
  - **Varianza del Laplaciano:** medida de nitidez. Disminuye al suavizar.
  - **Magnitud del gradiente de Sobel:** medida de la cantidad de bordes.

## 4. Resultado principal

El Gaussiano 5×5 (1/273) produce el mayor suavizado y el corte más marcado en las frecuencias altas. A cambio, es el que más detalle pierde. El Gaussiano 3×3 ofrece un compromiso intermedio, y el filtro de media es el que deja más ruido residual en la imagen con ruido.

## 5. Entorno

- Python 3.9 o superior, con NumPy, OpenCV y Matplotlib.
- Código de la práctica: *en preparación*.

## 6. Archivos de esta carpeta

- `readme.md`: este documento.
