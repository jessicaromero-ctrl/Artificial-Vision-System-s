"""Resume la exactitud y el F1 macro de la CNN entre semillas de entrenamiento.

Con n corridas, el IC 95 % de la media usa la distribución t de Student:
    media +/- t(0.975, n - 1) * sd / sqrt(n)

Uso:
    python seeds_summary.py --pattern "resultados/seed_*/metrics_cnn_resnet18.json"
"""

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t


def summarize(values, confidence=0.95):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    if n > 1:
        half = t.ppf(1 - (1 - confidence) / 2, n - 1) * sd / math.sqrt(n)
    else:
        half = float("nan")
    return {"n": n, "mean": mean, "sd": sd, "min": min(values), "max": max(values),
            "ci_low": mean - half, "ci_high": mean + half, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Resumen entre semillas de entrenamiento.")
    parser.add_argument("--pattern", default="resultados/seed_*/metrics_cnn_resnet18.json")
    parser.add_argument("--out", type=Path, default=Path("resultados/resumen_semillas.json"))
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No se encontraron métricas con el patrón: {args.pattern}")

    runs = []
    for f in files:
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        runs.append({"archivo": f, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "best_epoch": m.get("best_epoch")})

    result = {
        "runs": runs,
        "accuracy": summarize([r["accuracy"] for r in runs]),
        "macro_f1": summarize([r["macro_f1"] for r in runs]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nGuardado en: {args.out}")


if __name__ == "__main__":
    main()
