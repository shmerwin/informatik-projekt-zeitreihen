"""Trainiert eine robuste saisonale t-24-Baseline ohne Keras-Modell.

Aufruf aus dem Projektordner: ``python -m src.seasonal_baseline``
"""

import json

import numpy as np

from .benchmark_evaluation import _empirical_threshold
from .forecaster.common import (
    ALPHA, ANOMALY_FEATURE, DATA_PATH, FEATURES, ROOT, evaluation_metadata,
    fit_scaler, load_data, make_windows, scale_values, split_masks,
)


MODEL_DIR = ROOT / "saved_models" / "seasonal"


def _residuals(data, scaler):
    seasonal = scale_values(data["seasonal"], data["sensor_id"], scaler)
    return data["y_scaled"] - seasonal


def seasonal_scores(data, model, bundle):
    """Bewertet robuste t-24-Abweichungen, wenn beide Stunden gemessen sind."""
    valid = data["observed"] & data["seasonal_observed"]
    residual = (
        _residuals(data, bundle["scaler"])
        - np.asarray(bundle["residual_center"])
    ) / np.asarray(bundle["residual_scale"])
    target = FEATURES.index(bundle.get("anomaly_feature", ANOMALY_FEATURE))
    scores = np.abs(residual[:, target])
    scores[~valid] = np.nan
    return scores


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    scaler = fit_scaler(df)
    data = make_windows(df, scaler)
    masks = split_masks(data)
    valid_train = masks["train"] & data["seasonal_observed"]
    if not valid_train.any():
        raise ValueError("Keine gemessenen t-24-Trainingspaare vorhanden")

    residual = _residuals(data, scaler)[valid_train]
    center = np.median(residual, axis=0)
    scale = np.maximum(
        np.quantile(residual, 0.75, axis=0)
        - np.quantile(residual, 0.25, axis=0),
        0.05,
    )
    bundle = {
        "model": None,
        "features": FEATURES,
        "anomaly_feature": ANOMALY_FEATURE,
        "window_size": 24,
        "architecture": "robust seasonal t-24 residual",
        "parameters": 0,
        "scaler": scaler,
        "residual_center": center.tolist(),
        "residual_scale": scale.tolist(),
        "scorable_counts": {
            name: int((mask & data["observed"] & data["seasonal_observed"]).sum())
            for name, mask in masks.items()
        },
        **evaluation_metadata(data, masks),
    }
    scores = seasonal_scores(data, None, bundle)[masks["calibration"]]
    bundle["calibrator"] = {
        "alpha": ALPHA,
        "target_feature": ANOMALY_FEATURE,
        "threshold": _empirical_threshold(scores, ALPHA),
    }
    with open(MODEL_DIR / "model_bundle.json", "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saisonale Baseline gespeichert: {MODEL_DIR / 'model_bundle.json'}")
    return bundle


if __name__ == "__main__":
    train()
