"""Punto de entrada: ejecuta el baseline clásico o la CNN sobre el dataset NEU.

Ejemplos:
    python run_experiment.py --data-dir <ruta> --model baseline
    python run_experiment.py --data-dir <ruta> --model cnn --epochs 15 --img-size 224
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline import run_baseline  # noqa: E402
from cnn import run_cnn  # noqa: E402
from data import check_leakage, deduplicate, discover_samples, stratified_split  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Detección de defectos superficiales (NEU).")
    parser.add_argument("--data-dir", type=Path, required=True, help="Raíz del dataset descargado.")
    parser.add_argument("--model", choices=["baseline", "cnn"], required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=None,
                        help="Semilla del entrenamiento de la CNN; la partición sigue con --seed.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    samples, classes = discover_samples(args.data_dir)
    print(f"Imágenes encontradas: {len(samples)} | Clases ({len(classes)}): {classes}")

    samples, removed = deduplicate(samples)
    if removed:
        print(f"Duplicados eliminados antes del particionado: {len(removed)}")
        for path in removed:
            print(f"  - {path}")
    print(f"Imágenes únicas: {len(samples)}")

    train_samples, test_samples = stratified_split(samples, args.test_size, args.seed)
    print(f"Entrenamiento: {len(train_samples)} | Prueba: {len(test_samples)}")

    leaked = check_leakage(train_samples, test_samples)
    if leaked:
        raise SystemExit(
            f"Fuga de datos detectada: {len(leaked)} imágenes de prueba aparecen en entrenamiento."
        )
    print("Fuga de datos entre particiones: 0 (verificado por hash MD5)")

    if args.model == "baseline":
        metrics = run_baseline(train_samples, test_samples, classes, args.out_dir, args.seed)
    else:
        metrics = run_cnn(
            train_samples, test_samples, classes, args.out_dir,
            epochs=args.epochs, img_size=args.img_size,
            batch_size=args.batch_size, lr=args.lr,
            seed=args.train_seed if args.train_seed is not None else args.seed,
        )

    print("\nResultados en prueba:")
    summary = {k: metrics[k] for k in ("model", "accuracy", "macro_f1")}
    summary["ci95_accuracy"] = [
        round(metrics["confidence_intervals"]["accuracy"]["ci_low"], 4),
        round(metrics["confidence_intervals"]["accuracy"]["ci_high"], 4),
    ]
    summary["ci95_macro_f1"] = [
        round(metrics["confidence_intervals"]["macro_f1"]["ci_low"], 4),
        round(metrics["confidence_intervals"]["macro_f1"]["ci_high"], 4),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
