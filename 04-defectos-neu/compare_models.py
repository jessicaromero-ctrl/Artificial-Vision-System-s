"""Compara dos modelos sobre el MISMO conjunto de prueba.

Calcula la prueba exacta de McNemar y el IC bootstrap de la diferencia de exactitud.
Requiere haber ejecutado ambos modelos con los mismos datos y la misma semilla.

Ejemplo:
    python compare_models.py --res-dir resultados --a baseline_lbp_svm --b cnn_resnet18
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stats import mcnemar_exact, paired_bootstrap_diff  # noqa: E402


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Comparación estadística entre dos modelos.")
    parser.add_argument("--res-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--a", default="baseline_lbp_svm")
    parser.add_argument("--b", default="cnn_resnet18")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    rows_a = load_predictions(args.res_dir / f"predictions_{args.a}.csv")
    rows_b = load_predictions(args.res_dir / f"predictions_{args.b}.csv")

    paths_a = [r["path"] for r in rows_a]
    paths_b = [r["path"] for r in rows_b]
    if paths_a != paths_b:
        raise SystemExit("Los conjuntos de prueba no coinciden. Ejecuta ambos modelos con los mismos datos y semilla.")

    y_true = np.array([int(r["y_true"]) for r in rows_a])
    pred_a = np.array([int(r["y_pred"]) for r in rows_a])
    pred_b = np.array([int(r["y_pred"]) for r in rows_b])

    result = {
        "model_a": args.a,
        "model_b": args.b,
        "n_test": int(len(y_true)),
        "mcnemar": mcnemar_exact(y_true, pred_a, pred_b),
        "accuracy_difference": paired_bootstrap_diff(
            y_true, pred_a, pred_b, n_boot=args.n_boot
        ),
    }
    out = args.res_dir / f"comparacion_{args.a}_vs_{args.b}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()
