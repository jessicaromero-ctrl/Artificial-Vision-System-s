# Detección de defectos superficiales en acero (dataset NEU)

Clasificación de seis tipos de defecto superficial en láminas de acero laminado. Se comparan dos enfoques: un **baseline clásico de Procesamiento Digital de Imágenes (PDI)** con descriptores de textura LBP y un clasificador SVM, y una **red neuronal convolucional (CNN)** ResNet-18 con transfer learning.

## 1. Problemática

En la industria siderúrgica, los defectos superficiales (grietas, inclusiones, manchas, picaduras, escamas y rayas) reducen la calidad del producto, generan retrabajo y pueden provocar reclamaciones de clientes. La inspección manual es lenta, subjetiva y difícil de sostener en líneas de producción de alta velocidad. Un sistema automático que clasifique el tipo de defecto permite tomar acciones correctivas en el proceso productivo y no solo descartar piezas.

## 2. Dataset

- **Nombre:** NEU Surface Defect Database (publicado en Kaggle por `kaustubhdikshit`).
- **Clases (6):** `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- **Imágenes:** 1,800 en escala de grises, 300 por clase.
- **Organización:** `NEU-DET/{train,validation}/images/<clase>/`. La carpeta `annotations/` contiene XML de bounding boxes, que este proyecto no usa.

El dataset es originalmente de **detección de objetos**. En este proyecto se reutiliza para **clasificación** de la clase de defecto, tomando la clase desde el nombre de la carpeta.

## 3. Pipeline paso a paso

1. **Descarga:** `src/download.py` obtiene el dataset con `kagglehub` y lo guarda en la caché local (`~/.cache/kagglehub`), fuera del repositorio.
2. **Descubrimiento:** `discover_samples` recorre el directorio, asigna a cada imagen su clase (nombre de la carpeta padre) y verifica que existan las seis clases.
3. **Deduplicación:** `deduplicate` calcula el hash MD5 de cada archivo y elimina copias idénticas, conservando la primera. Se detectó una duplicada (`patches_101.jpg` y `patches_105.jpg`), que se excluye antes de particionar.
4. **Particionado:** `stratified_split` separa 80 % para entrenamiento y 20 % para prueba, manteniendo la proporción de clases y con semilla fija (42).
5. **Modelo baseline (PDI):** `src/baseline.py` extrae descriptores LBP uniformes en tres escalas, normaliza con `StandardScaler` y entrena un SVM con kernel RBF. Los hiperparámetros se buscan con validación cruzada de 5 particiones sobre el conjunto de entrenamiento.
6. **Modelo CNN:** `src/cnn.py` usa ResNet-18 preentrenada en ImageNet, con fine-tuning completo, aumentación de datos, AdamW y programación de tasa de aprendizaje por coseno. Una validación interna del 10 % del entrenamiento controla el *early stopping*.
7. **Evaluación:** `src/metrics.py` calcula exactitud, F1 macro, reporte por clase y matriz de confusión, y los guarda en `resultados/`.

## 4. Fundamentos técnicos

- **LBP (Local Binary Patterns):** para cada píxel compara su intensidad con la de sus vecinos y genera un código binario. El histograma de esos códigos describe la textura local. Con patrones *uniformes* se reduce la dimensionalidad sin perder la mayor parte de la información.
- **SVM con kernel RBF:** busca un hiperplano en un espacio transformado que separa las clases con el mayor margen posible. El kernel RBF permite fronteras no lineales.
- **CNN y transfer learning:** una red convolucional aprende jerarquías de características, desde bordes hasta patrones complejos. Partir de pesos preentrenados en ImageNet acelera el entrenamiento y permite buen desempeño con datasets pequeños. El *fine-tuning* ajusta todos los pesos al nuevo dominio.
- **Métricas:** la exactitud mide la proporción de aciertos. El F1 macro promedia el F1 de cada clase con el mismo peso, lo que evita que una clase dominante oculte errores en las demás.

## 5. Estructura del proyecto

```
04-defectos-neu/
├── README.md
├── requirements.txt
├── run_experiment.py      # punto de entrada
├── src/
│   ├── data.py            # descubrimiento, deduplicación y particionado
│   ├── metrics.py         # métricas y matriz de confusión
│   ├── baseline.py        # LBP + SVM
│   ├── cnn.py             # ResNet-18 con fine-tuning
│   └── download.py        # descarga desde Kaggle
└── resultados/            # métricas y matrices de confusión (pesos .pt excluidos)
```

## 6. Ejecución

Desde la carpeta `04-defectos-neu`, con el entorno virtual activo:

```bash
pip install -r requirements.txt
python src/download.py
python run_experiment.py --data-dir "<ruta_impresa_por_download>" --model baseline
python run_experiment.py --data-dir "<ruta_impresa_por_download>" --model cnn --epochs 15 --img-size 224
```

Argumentos principales: `--test-size` (0.2), `--seed` (42), `--epochs`, `--img-size`, `--batch-size`, `--lr`.

## 7. Resultados

### Baseline LBP + SVM

Conjunto de prueba: 360 imágenes (entrenamiento: 1,439; imágenes únicas: 1,799).

| Métrica | Valor |
|---|---|
| Exactitud (accuracy) | 0.9861 |
| F1 macro | 0.9861 |

El detalle por clase (precisión, recall y F1) y la matriz de confusión se encuentran en `resultados/metrics_baseline_lbp_svm.json` y `resultados/confusion_baseline_lbp_svm.png`.

### CNN ResNet-18

Pendiente de ejecución. Se reportará con el mismo conjunto de prueba y las mismas métricas para permitir la comparación directa.

## 8. Importancia de los resultados

Una exactitud de 98.6 % con un descriptor clásico indica que las clases del dataset son **texturalmente separables** con información local de bajo costo computacional. Esto tiene implicaciones prácticas: un sistema de este tipo puede ejecutarse en hardware modesto y su comportamiento es fácil de explicar a un ingeniero de calidad.

Al mismo tiempo, el resultado establece una **referencia exigente**. La CNN solo justifica su costo si supera de forma consistente al baseline, o si mejora en clases específicas donde el descriptor clásico confunde defectos.

## 9. Observaciones metodológicas y técnicas

- **Prevención de fuga de datos:** se eliminó la imagen duplicada antes de particionar. Si una copia hubiera quedado en entrenamiento y otra en prueba, las métricas estarían infladas.
- **Conjunto de prueba aislado:** no interviene en la búsqueda de hiperparámetros del baseline ni en la selección de épocas de la CNN. La validación se hace solo con datos de entrenamiento.
- **Reproducibilidad:** partición estratificada con semilla fija y resultados guardados en archivos.
- **Consistencia en la evaluación:** las etiquetas de texto (baseline) y de índice (CNN) pasan por la misma conversión antes de calcular métricas, para que ambos modelos se evalúen igual.
- **Reutilización del dataset de detección:** las imágenes provienen de un dataset de detección de objetos. Como la clase se toma de la carpeta, no se usan las bounding boxes. Esto es válido para clasificación, pero conviene documentarlo.

## 10. Limitaciones

- **Una sola partición de prueba:** con 360 imágenes, cada error pesa cerca de 0.28 puntos porcentuales. No se reportan intervalos de confianza ni validación cruzada externa.
- **Dominio controlado:** las imágenes son de laboratorio, con iluminación y encuadre uniformes. El desempeño en líneas reales puede ser menor.
- **Baseline con histograma global:** el descriptor LBP pierde información espacial, es decir, dónde ocurre el defecto dentro de la pieza.
- **Clasificación, no localización:** el sistema indica qué defecto hay, pero no dónde se encuentra.
- **Dependencia de la caché de Kaggle:** el dataset se descarga en la máquina de cada usuario; no está versionado en el repositorio.

## 11. Insights centrales

- Los descriptores de textura bien elegidos siguen siendo una opción fuerte para inspección superficial en entornos controlados.
- El punto de comparación justo para cualquier modelo de aprendizaje profundo en este caso es un baseline clásico bien ajustado, no un modelo aleatorio.
- La calidad del pipeline de datos (deduplicación, particionado, aislamiento del conjunto de prueba) afecta tanto la validez de los resultados como el modelo mismo.

## 12. Glosario

- **PDI (Procesamiento Digital de Imágenes):** conjunto de técnicas para transformar y analizar imágenes con algoritmos explícitos.
- **LBP (Local Binary Patterns):** descriptor de textura basado en comparaciones de intensidad entre un píxel y sus vecinos.
- **SVM (Support Vector Machine):** clasificador que maximiza el margen entre clases.
- **Kernel RBF:** función que permite fronteras de decisión no lineales en el SVM.
- **CNN (Red Neuronal Convolucional):** red que aprende filtros para extraer características espaciales de imágenes.
- **Transfer learning:** reutilizar un modelo entrenado en una tarea general para una tarea específica.
- **Fine-tuning:** ajuste de los pesos de un modelo preentrenado con datos del nuevo dominio.
- **Aumentación de datos:** generar variaciones de las imágenes (rotaciones, volteos, cambios de brillo) para mejorar la generalización.
- **Early stopping:** detener el entrenamiento cuando la validación deja de mejorar, para evitar sobreajuste.
- **Validación cruzada:** evaluar un modelo dividiendo los datos de entrenamiento en varias particiones rotativas.
- **Particionado estratificado:** dividir los datos manteniendo la proporción de clases en cada subconjunto.
- **Fuga de datos (data leakage):** información del conjunto de prueba que llega al entrenamiento y sobreestima el desempeño.
- **Hash MD5:** huella digital de un archivo, usada para detectar copias idénticas.
- **Exactitud (accuracy):** proporción de predicciones correctas.
- **Precisión (precision):** de las predicciones de una clase, proporción que es correcta.
- **Recall (sensibilidad):** de los casos reales de una clase, proporción que el modelo detecta.
- **F1:** media armónica entre precisión y recall.
- **F1 macro:** promedio del F1 de todas las clases, con el mismo peso por clase.
- **Matriz de confusión:** tabla que cruza clase real con clase predicha.
- **Sobreajuste (overfitting):** el modelo memoriza el entrenamiento y generaliza mal.

## 13. Aplicaciones futuras

- Detección y localización de defectos con modelos de detección de objetos, aprovechando las bounding boxes del dataset original.
- Validación con imágenes de líneas de producción reales y adaptación de dominio.
- Integración en un sistema de inspección en tiempo real con cámaras industriales.
- Análisis de tendencias de defectos para retroalimentar el proceso de laminación.
- Extensión a otros materiales y superficies (cerámica, madera, textiles).

## 14. Estado del proyecto

- [x] Descarga y verificación del dataset.
- [x] Deduplicación y particionado estratificado.
- [x] Baseline LBP + SVM (accuracy 0.9861, F1 macro 0.9861).
- [ ] CNN ResNet-18 con transfer learning.
- [ ] Comparación de modelos y análisis de errores.
