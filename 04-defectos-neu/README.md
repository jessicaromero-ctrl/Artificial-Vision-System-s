# Detección de defectos superficiales en acero (dataset NEU)

Clasificación de 6 tipos de defecto superficial en láminas de acero laminado,
comparando un baseline clásico de PDI (LBP + SVM) contra una CNN con transfer
learning (ResNet-18).

## Estructura

```
neu-surface-defect-vision/
├── data/              # dataset descargado (no versionar)
├── resultados/        # métricas, matrices de confusión y pesos (no versionar)
├── src/
│   ├── data.py        # descubrimiento, duplicados y particionado estratificado
│   ├── metrics.py     # métricas y matriz de confusión
│   ├── baseline.py    # LBP multiescala + SVM RBF
│   ├── cnn.py         # ResNet-18 con fine-tuning
│   └── download.py    # descarga desde Kaggle
├── run_experiment.py  # punto de entrada
└── requirements.txt
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

1. Configura las credenciales de Kaggle en `~/.kaggle/kaggle.json` (no las pegues en el chat ni en el repositorio).
2. Descarga el dataset:
   ```bash
   python src/download.py
   ```
   Copia la ruta impresa a `data/NEU`, o usa esa ruta directamente con `--data-dir`.
3. Ejecuta el baseline:
   ```bash
   python run_experiment.py --data-dir data/NEU --model baseline
   ```
4. Ejecuta la CNN:
   ```bash
   python run_experiment.py --data-dir data/NEU --model cnn --epochs 15 --img-size 224
   ```

## Métricas reportadas

- Exactitud (accuracy) y F1 macro sobre el conjunto de prueba.
- Precisión, recall y F1 por clase.
- Matriz de confusión por modelo.

## Notas metodológicas

- El particionado es estratificado y usa semilla fija (42) para que los resultados sean reproducibles.
- El conjunto de prueba no participa en el entrenamiento ni en la selección de épocas. La validación interna (10% del entrenamiento) solo se usa para early stopping.
- Se reporta el número de duplicados por hash MD5. Si hay duplicados entre particiones, las métricas están infladas.
