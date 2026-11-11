"""Métricas comunes y gráfica de matriz de confusión para ambos modelos."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_and_save(y_true, y_pred, classes, out_dir, name):
    """Calcula métricas, guarda JSON y matriz de confusión, y devuelve el diccionario."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

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

    metrics = {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": report,
    }
    (out_dir / f"metrics_{name}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
