# Detección de esquinas mediante el algoritmo de Harris

**Universidad Politécnica Metropolitana de Hidalgo**
Maestría en Ingeniería en Inteligencia Artificial

**Alumna:** Jessica Melani Romero Lora
**Profesor:** Dr. Ernesto García Amaro
**Fecha:** 23 de octubre de 2026

---

## Resumen

Se presenta un experimento de detección de esquinas con el algoritmo de Harris sobre seis imágenes en escala de grises. Se varió el umbral absoluto δ sobre la función de respuesta V en diez valores distribuidos logarítmicamente en el intervalo [900, 1 000 000]. Para cada combinación de imagen y umbral se registró el número de esquinas detectadas, N(δ). Los resultados muestran que N(δ) disminuye de forma monótona al aumentar δ en todas las imágenes, aunque la forma de la disminución depende del contenido de cada imagen.

## 1. Introducción

Una esquina es un punto de una imagen donde el brillo cambia de manera abrupta en dos direcciones. Su detección automática es un paso básico en visión por computadora, pues permite describir una escena mediante puntos estables y distinguibles. El detector de Harris es uno de los métodos clásicos para este fin.

El objetivo del experimento es evaluar el comportamiento del detector de Harris al variar el umbral absoluto δ sobre la función de respuesta de esquina. Se analiza cuántas esquinas se detectan en seis imágenes con características distintas y cómo cambia ese conteo al aumentar δ.

## 2. Fundamentos teóricos

El método parte de las derivadas parciales de la imagen. Sobre la imagen filtrada se calculan los gradientes con los operadores de Sobel, mediante los núcleos $G_x$ y $G_y$:

$$
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \qquad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I
$$

Con estos gradientes se construyen las componentes de la matriz de estructura:

$$
I_{xx} = G_x^2, \qquad I_{yy} = G_y^2, \qquad I_{xy} = G_x \, G_y
$$

Cada componente se suaviza con un núcleo gaussiano de 3 × 3, obteniendo $A$, $B$ y $C$:

$$
A = g * I_{xx}, \qquad B = g * I_{yy}, \qquad C = g * I_{xy}
$$

La función de respuesta de Harris se define como:

$$
V = (A \cdot B - C^2) - \alpha \, (A + B)^2, \qquad \alpha = 0.04
$$

El término $A \cdot B - C^2$ corresponde al determinante de la matriz de estructura, y $(A + B)$ a su traza. Valores grandes de $V$ indican esquinas; valores negativos, bordes; y valores pequeños, regiones planas.

### 2.1 El umbral δ

El parámetro δ no es una desviación estándar ni un parámetro del suavizado. Es un umbral absoluto sobre $V$. Un punto se considera esquina cuando es máximo local de $V$ en una vecindad de 3 × 3 y su valor satisface $V \geq \delta$:

$$
\text{esquina}(x,y) \iff V(x,y) = \max_{3\times 3} V \ \land \ V(x,y) \geq \delta
$$

El número de esquinas para un valor dado de δ es entonces:

$$
N(\delta) = \sum_{x,y} \mathbf{1}\left[ V(x,y) = \max_{3\times 3} V \ \land \ V(x,y) \geq \delta \right]
$$

## 3. Metodología

### 3.1 Imágenes

Se emplearon seis imágenes en escala de grises: Cuadrado, Cuadrícula, Estrella, Figuras, Lena_Gray y Libros. Cada una presenta una complejidad estructural distinta, lo que permite contrastar el comportamiento del detector.

### 3.2 Procedimiento

Para cada imagen se ejecutó el siguiente procedimiento:

1. Filtrado de media 3 × 3 con núcleo de valores 1/9, con borde igual a cero.
2. Cálculo de gradientes con los núcleos de Sobel 3 × 3.
3. Construcción de las componentes $I_{xx}$, $I_{yy}$ e $I_{xy}$ de la matriz de estructura.
4. Suavizado gaussiano 3 × 3 de cada componente.
5. Cálculo de la respuesta $V$ de Harris, con $\alpha = 0.04$.
6. Supresión de no máximos mediante dilatación 3 × 3.
7. Aplicación del umbral δ y conteo de esquinas.

### 3.3 Valores de δ

El umbral se evaluó en diez valores distribuidos logarítmicamente sobre el intervalo [900, 1 000 000]. La distribución logarítmica asigna más valores a la zona baja del rango, donde se concentran los cambios más pronunciados del conteo. Cada valor se obtiene multiplicando el anterior por un factor constante $r$:

$$
\delta_k = \delta_{\min} \cdot r^{k}, \qquad r = \left(\frac{\delta_{\max}}{\delta_{\min}}\right)^{\frac{1}{N-1}}, \qquad k = 0, 1, \ldots, N-1
$$

Con $\delta_{\min} = 900$, $\delta_{\max} = 1\,000\,000$ y $N = 10$, el factor resulta $r \approx 2.18$.

### 3.4 Visualización

Las esquinas puntuales se marcan como píxeles individuales. Las esquinas dilatadas se obtienen ensanchando cada punto con un núcleo de 5 × 5, para facilitar su lectura en figuras de baja resolución. Ambas se presentan superpuestas sobre la imagen original y en máscara binaria.

## 4. Resultados

La Tabla 1 muestra el número de esquinas $N(\delta)$ detectadas en cada imagen para los diez valores de umbral.

**Tabla 1.** Número de esquinas detectadas $N(\delta)$ por imagen y valor de umbral δ.

| δ | Cuadrado | Cuadrícula | Estrella | Figuras | Lena_Gray | Libros |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 900 | 66 | 8 266 | 497 | 1 317 | 6 975 | 19 342 |
| 1 961 | 60 | 8 141 | 277 | 915 | 5 833 | 14 439 |
| 4 276 | 51 | 8 032 | 152 | 637 | 4 897 | 10 203 |
| 9 321 | 46 | 7 922 | 96 | 462 | 4 047 | 7 034 |
| 20 319 | 39 | 7 777 | 62 | 343 | 3 348 | 4 827 |
| 44 292 | 38 | 7 490 | 51 | 258 | 2 792 | 3 316 |
| 96 548 | 38 | 7 108 | 46 | 215 | 2 278 | 2 380 |
| 210 457 | 38 | 6 558 | 41 | 184 | 1 850 | 1 668 |
| 458 756 | 38 | 5 861 | 34 | 158 | 1 484 | 1 259 |
| 1 000 000 | 38 | 5 118 | 29 | 138 | 1 210 | 985 |

Los conteos se obtuvieron con el script del experimento y se presentan sin redondeo.

## 5. Análisis

Para comparar la sensibilidad de cada imagen al umbral se calculó la reducción relativa del conteo entre el valor mínimo y el máximo de δ:

$$
R = \frac{N(\delta_{\min}) - N(\delta_{\max})}{N(\delta_{\min})} \times 100
$$

La razón entre ambos conteos, $N(\delta_{\min}) / N(\delta_{\max})$, indica cuántas veces disminuye el número de esquinas en el intervalo.

**Tabla 2.** Reducción relativa y razón de conteos entre δ = 900 y δ = 1 000 000.

| Imagen | N(900) | N(1 000 000) | Reducción R (%) | Razón |
| :--- | :---: | :---: | :---: | :---: |
| Cuadrado | 66 | 38 | 42.4 | 1.7 |
| Cuadrícula | 8 266 | 5 118 | 38.1 | 1.6 |
| Estrella | 497 | 29 | 94.2 | 17.1 |
| Figuras | 1 317 | 138 | 89.5 | 9.5 |
| Lena_Gray | 6 975 | 1 210 | 82.7 | 5.8 |
| Libros | 19 342 | 985 | 94.9 | 19.6 |

### 5.1 Comportamiento por imagen

**Cuadrado.** El conteo desciende de 66 a 38 esquinas y se estabiliza a partir de δ ≈ 44 292. Las esquinas que permanecen corresponden al cuadro central y a los vértices exteriores de la figura.

**Cuadrícula.** Es la imagen con mayor número de esquinas. La reducción es gradual, ya que cada cruce de la malla constituye una esquina y el umbral debe aumentar considerablemente para disminuir el conteo.

**Estrella.** El conteo baja de 497 a 29. Al inicio se marca el contorno completo de las cinco puntas; al aumentar δ desaparecen primero los detalles de menor respuesta y permanecen los vértices principales.

**Figuras.** El descenso es continuo, de 1 317 a 138. Con umbrales bajos se marcan los bordes de los sólidos y sus sombras; con umbrales altos sobreviven los vértices de las piezas de mayor tamaño.

**Lena_Gray.** El conteo pasa de 6 975 a 1 210. Con umbrales bajos se marca gran parte de la textura del cabello, el fondo y las paredes. Al aumentar δ, las esquinas se concentran en el sombrero, los ojos, la nariz y el contorno del hombro.

**Libros.** Es la imagen con mayor razón de reducción, de 19 342 a 985. Su gran cantidad de líneas y texturas genera numerosos candidatos a esquina, de los cuales sobreviven, al final, los bordes de los libros apilados.

## 6. Conclusiones

1. El umbral δ regula la selectividad del detector de Harris. En las seis imágenes, $N(\delta)$ disminuye de forma monótona al aumentar δ, como corresponde a un criterio de corte absoluto.

2. La forma de la disminución depende del contenido de la imagen. Las figuras simples, como Cuadrado y Estrella, presentan pocas esquinas y una estabilización temprana o una caída pronunciada. Las imágenes con estructura repetitiva, como Cuadrícula, requieren umbrales muy altos para reducir su conteo de forma apreciable.

3. Las imágenes con textura, como Lena_Gray y Libros, generan numerosos candidatos a esquina con umbrales bajos. El aumento de δ filtra esos candidatos y conserva los rasgos de mayor respuesta, como los bordes de los objetos y los rasgos faciales.

4. El conteo, por sí solo, no valida la calidad de las detecciones. La comparación visual entre las figuras superpuestas es necesaria para verificar que las esquinas marcadas corresponden a estructuras relevantes de la escena.

5. Para una aplicación concreta, la elección de δ debe considerar tanto el número de esquinas deseado como el tipo de contenido de la imagen. Un mismo valor de umbral no produce el mismo nivel de selectividad en escenas distintas.

## 7. Actividades realizadas

1. Revisión del algoritmo de Harris visto en clase y de su código de referencia.
2. Adaptación del código en `harrys.py`, con δ definido como umbral absoluto sobre $V$.
3. Configuración del entorno: verificación de Miniconda y de las librerías `numpy`, `opencv-python` y `matplotlib`.
4. Ejecución del script desde VS Code sobre las seis imágenes de la carpeta `ejercicio`.
5. Obtención de los conteos $N(\delta)$ y generación de las figuras en `resultados_grid/`.
6. Cálculo de la reducción relativa y de la razón entre conteos (Tabla 2).
7. Elaboración del reporte con la descripción del método, los resultados y las conclusiones.

## 8. Archivos generados

| Archivo | Contenido |
| :--- | :--- |
| `tabla_conteo_10_deltas_log.md` | Tabla de conteos (Tabla 1) |
| `plot_puntual_overlay_log.png` | Esquinas puntuales superpuestas sobre la imagen |
| `plot_puntual_binary_log.png` | Esquinas puntuales en máscara binaria |
| `plot_dilatada_overlay_log.png` | Esquinas dilatadas superpuestas sobre la imagen |
| `plot_dilatada_binary_log.png` | Esquinas dilatadas en máscara binaria |

## 9. Ejecución

```bash
python harrys.py
```

Los parámetros del experimento son: $\alpha = 0.04$, $\delta \in [900, 1\,000\,000]$, $N = 10$ valores en escala logarítmica, y dilatación de 5 × 5 para las esquinas dilatadas.
