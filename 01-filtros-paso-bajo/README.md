# Sistemas de Visión Artificial: Prácticas

| | |
|---|---|
| **Programa** | Maestría en Inteligencia Artificial |
| **Institución** | Universidad Politécnica Metropolitana del Estado de Hidalgo (UPMH) |
| **Asignatura** | Sistemas de Visión Artificial |
| **Profesor** | Dr. Ernesto García Amaro |
| **Alumna** | Jessica Melani Romero Lora |
| **Periodo** | 2026 |

---

> ⚠️ **AVISO DE USO ACADÉMICO Y DE DATOS**
>
> - Este repositorio contiene **prácticas de la Maestría en Inteligencia Artificial de la UPMH**. Su contenido tiene **fines exclusivamente académicos y educativos**.
> - La práctica 2 usa **fotografías clínicas de lesiones cutáneas de pacientes con neurofibromatosis tipo 1 (NF1)**. Estas imágenes **no se incluyen en este repositorio**. No deben redistribuirse ni publicarse sin autorización de la institución y de las personas involucradas.
> - Los documentos y el código están **protegidos como material académico**. Se prohíbe su reproducción o distribución total o parcial sin autorización de la autora.

---

## Contenido

```
Artificial-Vision-System-s/
├── README.md
├── 01-filtros-paso-bajo/
│   └── Reporte_Filtros_Paso_Bajo.pdf
└── 02-operaciones-pixeles/
    ├── Reporte_Operaciones_Pixeles.pdf
    ├── practica_operaciones_pixeles.py
    └── comparativo_original_vs_procesado (nombre completo por confirmar)
```

| Práctica | Carpeta | Documento principal | Código |
|---|---|---|---|
| 1. Análisis comparativo de filtros paso bajo espaciales | `01-filtros-paso-bajo/` | `Reporte_Filtros_Paso_Bajo.pdf` | *(pendiente de incluir)* |
| 2. Operaciones aritméticas y lógicas a nivel de píxel | `02-operaciones-pixeles/` | `Reporte_Operaciones_Pixeles.pdf` | `practica_operaciones_pixeles.py` |

---

## 1. Filtros paso bajo espaciales

Compara tres filtros de promediado y suavizado aplicados por convolución discreta:

- **Media 1/9:** kernel 3×3 con pesos uniformes.
- **Gaussiano 1/16:** kernel 3×3 con pesos binomiales `[1 2 1; 2 4 2; 1 2 1]`.
- **Gaussiano 1/273:** kernel 5×5 con pesos binomiales, normalizado por su suma (273).

**Análisis:**

- Respuesta en frecuencia de cada filtro, en una dimensión y en dos dimensiones, y ganancia en dB.
- Efecto sobre cuatro imágenes de prueba: Lena (RGB), Lena en escala de grises, paisaje y paisaje con ruido.
- Métricas de calidad: **MSE/PSNR** (distancia respecto a la original), **SSIM** (similitud estructural), **varianza del Laplaciano** (medida de nitidez) y **magnitud del gradiente de Sobel** (bordes).

**Resultado principal:** el Gaussiano 1/273 produce el mayor suavizado y el corte más marcado en frecuencias altas, a cambio de perder más detalle.

*[Pendiente de incluir: el código que genera las figuras y métricas del reporte.]*

## 2. Operaciones aritméticas y lógicas a nivel de píxel

Aplica las operaciones **SUMA**, **RESTA**, **AND** y **OR** entre imágenes, sobre un dataset de **7 imágenes dermatológicas de 512×512 píxeles**. Las imágenes se clasifican en máculas y nódulos.

- **Implementación:** Python 3.9 o superior, con NumPy, OpenCV y Matplotlib.
- **Código:** `practica_operaciones_pixeles.py`.
- **Comparativo:** documento que contrasta las imágenes originales con las resultantes.

### Requisitos

```bash
pip install numpy opencv-python matplotlib
```

### Ejecución

```bash
python practica_operaciones_pixeles.py
```

*[Pendiente de completar: ruta en la que el script espera las imágenes del dataset, y la descripción de los parámetros de entrada y salida.]*

---

## Estado del repositorio

- Reportes en PDF de ambas prácticas.
- Código de la práctica 2.
- Código de la práctica 1: pendiente.
- Dataset de imágenes: no incluido por razones de privacidad.
