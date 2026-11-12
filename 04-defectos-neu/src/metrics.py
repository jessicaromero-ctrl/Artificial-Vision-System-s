"""Métricas, intervalos de confianza y registro de predicciones para ambos modelos."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from stats import bootstrap_ci  # noqa: E402


def to_indices(values, classes):
    """Convierte etiquetas (nombres de clase o índices) a índices enteros."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array(
        [class_to_idx[v] if isinstance(v, str) else int(v) for v in values],
        dtype=int,
    )


def evaluate_and_save(y_true, y_pred, classes, out_dir, name, paths=None,
                      n_boot=2000, seed=42):
    """Calcula métricas con IC 95 %, guarda JSON, predicciones y matriz de confusión."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = to_indices(y_true, classes)
    y_pred = to_indices(y_pred, classes)
    labels = list(range(len(classes)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    ci_acc = bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=n_boot, seed=seed)
    ci_f1 = bootstrap_ci(y_true, y_pred, metric="macro_f1", labels=labels,
                         n_boot=n_boot, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión: {name}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{name}.png", dpi=150)
    plt.close(fig)

    if paths is not None:
        with open(out_dir / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "y_true", "y_pred", "y_true_name", "y_pred_name"])
            for p, t, q in zip(paths, y_true, y_pred):
                writer.writerow([str(p), int(t), int(q), classes[t], classes[q]])

    metrics = {
        "model": name,
        "n_test": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confidence_intervals": {"accuracy": ci_acc, "macro_f1": ci_f1},
        "per_class": report,
    }
    (out_dir / f"metrics_{name}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
