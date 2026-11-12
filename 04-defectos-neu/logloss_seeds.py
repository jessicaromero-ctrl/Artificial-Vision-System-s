"""Calcula la pérdida de entropía cruzada (log-loss) en prueba para cada corrida por semilla.

No reentrena: carga el best_resnet18.pt de cada carpeta resultados/seed_*/ y evalúa sobre el
mismo conjunto de prueba (partición con semilla 42). La log-loss es continua, así que sigue
distinguiendo modelos aun cuando la exactitud está saturada en 1.0.

Notación:
    NLL_i = -log p_{i, y_i},    log-loss = (1/n) sum_i NLL_i

Uso:
    python logloss_seeds.py --data-dir <ruta> --img-size 224
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cnn import build_model, build_transforms  # noqa: E402
from data import deduplicate, discover_samples, stratified_split  # noqa: E402
from seeds_summary import summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Log-loss en prueba por semilla.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición de prueba.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--pattern", default="resultados/seed_*/best_resnet18.pt")
    parser.add_argument("--out", type=Path, default=Path("resultados/logloss_semillas.json"))
    args = parser.parse_args()

    samples, classes = discover_samples(args.data_dir)
    samples, _ = deduplicate(samples)
    _, test = stratified_split(samples, args.test_size, args.seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths = [p for p, _ in test]
    y = np.array([class_to_idx[l] for _, l in test])
    _, eval_tf = build_transforms(args.img_size)
    batch = 32

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No se encontraron pesos con el patrón: {args.pattern}")

    runs = []
    for f in files:
        model = build_model(len(classes))
        model.load_state_dict(torch.load(f, map_location="cpu"))
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                xb = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
                probs.append(torch.softmax(model(xb), dim=1).numpy())
        probs = np.concatenate(probs)
        nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))
        runs.append({
            "archivo": f,
            "accuracy": float((probs.argmax(1) == y).mean()),
            "logloss_test": float(nll.mean()),
        })
        print(f"{f}: accuracy={runs[-1]['accuracy']:.4f} logloss={runs[-1]['logloss_test']:.5f}")

    result = {
        "runs": runs,
        "logloss_test": summarize([r["logloss_test"] for r in runs]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["logloss_test"], indent=2))
    print(f"\nGuardado en: {args.out}")


if __name__ == "__main__":
    main()
