# Operaciones aritméticas y lógicas a nivel de píxel

| | |
|---|---|
| **Asignatura** | Sistemas de Visión Artificial |
| **Programa** | Maestría en Inteligencia Artificial |
| **Institución** | Universidad Politécnica Metropolitana del Estado de Hidalgo (UPMH) |
| **Profesor** | Dr. Ernesto García Amaro |
| **Alumna** | Jessica Melani Romero Lora |
| **Periodo** | 2026 |
| **Actividad** | 2. Operaciones a nivel de píxel |

---

> ⚠️ **AVISO DE USO ACADÉMICO Y DE DATOS**
>
> - Esta práctica forma parte de la **Maestría en Inteligencia Artificial de la UPMH** y tiene **fines exclusivamente académicos y educativos**.
> - Las imágenes son **fotografías clínicas de lesiones cutáneas de pacientes con neurofibromatosis tipo 1 (NF1)**. **No se incluyen en este repositorio** y no deben redistribuirse ni publicarse sin autorización de la institución y de las personas involucradas.
> - El contenido está **protegido como material académico**. Se prohíbe su reproducción o distribución total o parcial sin autorización de la autora.

---

## 1. Descripción

La práctica aplica operaciones aritméticas y lógicas entre imágenes, a nivel de píxel, para comparar imágenes originales con los resultados de cada operación.

## 2. Operaciones

| Operación | Descripción |
|---|---|
| **SUMA** | Suma los valores de los píxeles correspondientes de dos imágenes. |
| **RESTA** | Resta los valores de los píxeles correspondientes. Resalta las diferencias entre dos imágenes. |
| **AND** | Operación lógica bit a bit. Conserva los bits comunes a ambas imágenes. |
| **OR** | Operación lógica bit a bit. Conserva los bits presentes en cualquiera de las dos imágenes. |

## 3. Dataset

Siete imágenes dermatológicas, con lesiones de tipo **máculas** y **nódulos** asociadas a NF1.

| Tipo | Imágenes |
|---|---|
| Máculas | 1, 2, 4, 6 |
| Nódulos | 3, 5, 7 |

**Preprocesamiento.** Las imágenes originales tenían resoluciones distintas: entre 608 y 790 px de ancho, y entre 468 y 606 px de alto. Todas se redimensionaron a **512×512 píxeles** antes de aplicar las operaciones. En la tabla de la práctica se registra el valor medio de los canales B, G y R de cada imagen.

## 4. Requisitos

- Python 3.9 o superior.
- Librerías: NumPy, OpenCV (`opencv-python`) y Matplotlib.

```bash
pip install numpy opencv-python matplotlib
```

## 5. Ejecución

```bash
python practica_operaciones_pixeles.py
```

## 6. Salidas

- Comparativo entre las imágenes originales y el resultado de cada operación.

## 7. Archivos de esta carpeta

- `readme.md`: este documento.
- `practica_operaciones_pixeles.py`: código de la práctica. *(Debe colocarse en esta misma carpeta.)*
