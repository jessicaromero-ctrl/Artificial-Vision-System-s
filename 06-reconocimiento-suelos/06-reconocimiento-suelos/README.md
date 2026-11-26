# Tumor Risk Modeling for Patients with Neurofibromatosis Type 1 (NF1) Using Machine Learning Algorithms

**Ingeniería de Clasificación de Riesgo en Neurofibromatosis Tipo 1 (NF1) basada en Características Clínicas mediante Ensemble Learning**

Autora: **Jessica Melani Romero Lora**

---

## Sobre el proyecto

La Neurofibromatosis Tipo 1 (NF1) es una enfermedad genética rara que predispone a complicaciones tumorales de gravedad variable, lo que hace que la identificación temprana de pacientes de alto riesgo sea crítica para priorizar recursos clínicos limitados. Este proyecto desarrolla un modelo de *machine learning* que clasifica el riesgo de complicación tumoral severa (`Tumour_Case`) a partir de características clínicas de pacientes con NF1.

El objetivo no es solo maximizar la precisión predictiva, sino construir un modelo **robusto y validado rigurosamente**, cuyo desempeño pueda sostenerse ante el desafío que representan los datos escasos propios de una enfermedad rara — un paso necesario antes de pensar en su aplicación clínica real y en el acceso equitativo a la atención especializada que motivó este trabajo.

## Metodología

El pipeline está organizado en cuatro fases secuenciales:

### Fase 1 — Preprocesamiento y limpieza
- Estandarización de nombres de columnas (eliminación de caracteres especiales y espacios).
- Renombrado de variables clave para asegurar consistencia (`Tumour_Case`, `Cafe_au_Lait`).
- División estratificada train/test (80/20) para preservar el balance de clases.
- Análisis de correlación (umbral |0.9|) para detectar redundancia entre variables.

### Fase 2 — Feature Engineering
Selección de variables mediante dos enfoques complementarios, cuya **intersección** define el subconjunto final de features:
- **Regularización L1 (LASSO)** vía `LogisticRegression(penalty='l1')`, con búsqueda del mejor hiperparámetro `C` por `GridSearchCV`.
- **Importancia de características por Random Forest**, seleccionando variables con importancia por encima de la media.

### Fase 3 — Modelado y validación
- Validación cruzada estratificada (`StratifiedKFold`, K=5) para garantizar estabilidad frente al desbalance de clases.
- Comparación de tres modelos mediante *ensemble learning*:
  - Regresión Logística (RL)
  - Random Forest (RF)
  - XGBoost
- Optimización de hiperparámetros por modelo mediante `GridSearchCV` (scoring: `roc_auc`).
- Evaluación fold por fold, reentrenando un modelo limpio (`clone`) en cada partición para evitar fuga de información.

### Fase 4 — Métricas
Comparación final de la estabilidad de los tres modelos mediante:
- **AUC** (capacidad discriminativa)
- **F1-Score** (balance precisión/recall)
- **Recall** (sensibilidad — particularmente relevante dado el costo clínico de no detectar un caso de riesgo alto)

Cada métrica se reporta como **media y desviación estándar** a través de los 5 folds, permitiendo comparar no solo el desempeño promedio de cada modelo, sino su estabilidad.

## Estructura del pipeline

```
Datos crudos (CSV)
    │
    ▼
Preprocesamiento y limpieza ──▶ División estratificada 80/20
    │
    ▼
Feature Engineering (LASSO ∩ Random Forest)
    │
    ▼
Validación cruzada estratificada (K=5)
    │
    ├── Regresión Logística
    ├── Random Forest
    └── XGBoost
    │
    ▼
Tabla comparativa de estabilidad (AUC / F1 / Recall)
```

## Dataset

Los datos utilizados corresponden a características clínicas de pacientes con NF1, con `Tumour_Case` como variable objetivo (complicación tumoral severa) y `CaseType` excluida del set de features. El dataset se carga directamente desde GitHub en formato CSV.

> **Nota:** por tratarse de datos clínicos, se recomienda verificar la fuente y los términos de uso del dataset antes de su reutilización.

## Tecnologías

| Categoría | Herramientas |
|---|---|
| Manipulación de datos | `pandas`, `numpy` |
| Visualización | `matplotlib`, `seaborn` |
| Modelado | `scikit-learn`, `xgboost` |
| Validación | `StratifiedKFold`, `GridSearchCV` |

## Instalación

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Uso

El notebook/script principal (`nf1_py.py`, originalmente desarrollado en Google Colab) ejecuta el pipeline completo de principio a fin:

```bash
python nf1_py.py
```

Al finalizar, se generan tres salidas principales:
1. Tabla de resultados por fold y modelo.
2. Tabla comparativa de estabilidad (media y desviación estándar de AUC, F1 y Recall).
3. Visualización de la matriz de correlación entre variables clínicas.

## Próximos pasos

Este modelo representa una primera iteración validada. Las siguientes líneas de mejora están en desarrollo:
- Incorporación de datos de imagen (MRI) y datos longitudinales para fortalecer la validación clínica.
- Evaluación de la generalización del modelo entre distintas poblaciones de pacientes.
- Comunicación de la incertidumbre del modelo de forma interpretable para el personal clínico.
- Validación externa con un partner clínico e institucional.

## Autora

**Jessica Melani Romero Lora**
[LinkedIn](https://www.linkedin.com/in/jessica-romero-lora)

Perfil híbrido en sociología y desarrollo técnico en machine learning, con foco en el uso de IA para el acceso equitativo a la salud pública.

---

*Proyecto desarrollado en el marco de la iniciativa AI for Health (Universidad de Groningen – UNAM).*
