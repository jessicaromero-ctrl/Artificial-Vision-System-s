"""Baseline clásico (PDI): descriptores LBP multiescala + SVM con kernel RBF.

Sirve como referencia para comparar contra la CNN con transfer learning.
"""

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import evaluate_and_save

IMG_SIZE = 200
# (P vecinos, R radio). Varias escalas capturan textura fina y gruesa.
LBP_CONFIG = [(8, 1), (16, 2), (24, 3)]


def lbp_features(path, img_size=IMG_SIZE):
    img = np.asarray(
        Image.open(path).convert("L").resize((img_size, img_size)), dtype=np.uint8
    )
    feats = []
    for n_points, radius in LBP_CONFIG:
        codes = local_binary_pattern(img, n_points, radius, method="uniform")
        n_bins = n_points + 2  # patrones uniformes + una clase para el resto
        hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins), density=True)
        feats.append(hist)
    return np.concatenate(feats)


def extract(samples):
    X = np.stack([lbp_features(p) for p, _ in samples])
    y = np.array([l for _, l in samples])
    return X, y


def run_baseline(train_samples, test_samples, classes, out_dir, seed=42):
    X_train, y_train = extract(train_samples)
    X_test, y_test = extract(test_samples)

    pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed))
    search = GridSearchCV(
        pipeline,
        param_grid={"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
        cv=5,
        n_jobs=-1,
        scoring="f1_macro",
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    metrics = evaluate_and_save(
        y_test, y_pred, classes, out_dir, name="baseline_lbp_svm",
        paths=[path for path, _ in test_samples],
    )
    metrics["best_params"] = search.best_params_
    return metrics
