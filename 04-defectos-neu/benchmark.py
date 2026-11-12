"""Mide latencia por imagen, memoria pico y tamaño en disco de un modelo entrenado.

Cada modelo se mide en una invocación separada para aislar la memoria del proceso.
La latencia es extremo a extremo: lectura del archivo, preprocesamiento e inferencia,
una imagen por llamada, en CPU. Se descartan 10 llamadas de calentamiento.

Ejemplos:
    python benchmark.py --data-dir <ruta> --model baseline
    python benchmark.py --data-dir <ruta> --model cnn --img-size 128

Salida: resultados/benchmark_<modelo>.json
"""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import deduplicate, discover_samples, stratified_split  # noqa: E402


def peak_rss_mb():
    """Memoria máxima residente del proceso. macOS reporta bytes; Linux, kilobytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 ** 2 if platform.system() == "Darwin" else 1024
    return ru / divisor


def current_rss_mb():
    return psutil.Process().memory_info().rss / 1024 ** 2


def time_per_image(predict_one, paths, warmup=10):
    for p in paths[:warmup]:
        predict_one(p)
    times = []
    for p in paths:
        t0 = time.perf_counter()
        predict_one(p)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {
        "n_timed": int(len(t)),
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
    }


def bench_baseline(test_paths, res_dir):
    import joblib
    from baseline import lbp_features

    model_path = res_dir / "best_svm_baseline.joblib"
    if not model_path.exists():
        raise SystemExit(f"No existe {model_path}. Ejecuta primero el baseline con run_experiment.py.")

    rss_start = current_rss_mb()
    model = joblib.load(model_path)
    rss_loaded = current_rss_mb()

    def predict_one(p):
        return model.predict(lbp_features(p).reshape(1, -1))

    latency = time_per_image(predict_one, test_paths)
    svc = model.named_steps["svc"]
    return {
        "model": "baseline_lbp_svm",
        "disk_size_mb": model_path.stat().st_size / 1024 ** 2,
        "support_vectors": int(svc.n_support_.sum()),
        "feature_dim": int(lbp_features(test_paths[0]).shape[0]),
        "rss_start_mb": rss_start,
        "rss_after_load_mb": rss_loaded,
        "latency": latency,
    }


def bench_cnn(test_paths, res_dir, img_size, classes):
    import torch
    from PIL import Image
    from cnn import build_model, build_transforms

    weights_path = res_dir / "best_resnet18.pt"
    if not weights_path.exists():
        raise SystemExit(f"No existe {weights_path}. Ejecuta primero la CNN con run_experiment.py.")

    _, eval_tf = build_transforms(img_size)
    rss_start = current_rss_mb()
    model = build_model(len(classes))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    rss_loaded = current_rss_mb()

    def predict_one(p):
        x = eval_tf(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return model(x).argmax(1)

    latency = time_per_image(predict_one, test_paths)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": "cnn_resnet18",
        "img_size": img_size,
        "parameters": int(n_params),
        "disk_size_mb": weights_path.stat().st_size / 1024 ** 2,
        "rss_start_mb": rss_start,
        "rss_after_load_mb": rss_loaded,
        "latency": latency,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark de latencia y memoria.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["baseline", "cnn"], required=True)
    parser.add_argument("--res-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=128,
                        help="Debe coincidir con el entrenamiento de la CNN.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Usar solo las primeras N imágenes de prueba (0 = todas).")
    return parser.parse_args()


def main():
    args = parse_args()
    samples, classes = discover_samples(args.data_dir)
    samples, _ = deduplicate(samples)
    _, test = stratified_split(samples, args.test_size, args.seed)
    test_paths = [p for p, _ in test]
    if args.limit:
        test_paths = test_paths[: args.limit]

    args.res_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "baseline":
        result = bench_baseline(test_paths, args.res_dir)
    else:
        result = bench_cnn(test_paths, args.res_dir, args.img_size, classes)

    result["peak_rss_mb"] = peak_rss_mb()
    result["platform"] = f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"
    result["latency_definition"] = "extremo a extremo por imagen, CPU, incluye lectura y preprocesamiento"

    suffix = f"_{args.img_size}px" if args.model == "cnn" else ""
    out = args.res_dir / f"benchmark_{result['model']}{suffix}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()
