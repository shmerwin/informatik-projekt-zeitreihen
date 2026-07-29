"""Trainiert einen kleinen LSTM-Autoencoder für 24-Stunden-Fenster.

Aufruf aus dem Projektordner: ``python -m src.autoencoder.train``
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from ..forecaster.common import (
    ALPHA, ANOMALY_FEATURE, CALIBRATION_END, DATA_PATH, FEATURES, SEED, TRAIN_END,
    TUNE_END, WINDOW_SIZE, fit_scaler, load_data, make_windows, set_seed,
    split_masks,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "saved_models" / "autoencoder"
EPOCHS = 100
BATCH_SIZE = 128
MAX_MASK_HOURS = 4
LSTM_UNITS = 16


def sequences(data):
    """Verbindet 24 Kontextstunden mit der zu rekonstruierenden Zielstunde."""
    current = np.concatenate([
        data["y_scaled"][:, None, :],
        data["observed"][:, None, None].astype("float32"),
    ], axis=2)
    return np.concatenate([data["history"][:, 1:, :3], current], axis=1)


def add_masking_noise(x, seed=SEED):
    """Verdeckt einen gemessenen Block von höchstens vier Stunden."""
    noisy = x.copy()
    rng = np.random.default_rng(seed)
    for number, window in enumerate(noisy):
        length = int(rng.integers(1, MAX_MASK_HOURS + 1))
        starts = [
            i for i in range(WINDOW_SIZE - length + 1)
            if x[number, i:i + length, 2].all()
        ]
        if starts:
            start = int(rng.choice(starts))
            window[start:start + length, :2] = 0
            window[start:start + length, 2] = 0
    return noisy


def build_model():
    """Bewusst kompakter Encoder-Decoder nach dem klassischen LSTM-AE-Prinzip."""
    model = models.Sequential([
        layers.Input((WINDOW_SIZE, 3)),
        layers.LSTM(LSTM_UNITS),
        layers.RepeatVector(WINDOW_SIZE),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.Dense(len(FEATURES)),
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    return model


def reconstruction_errors(model, data):
    x = sequences(data)
    reconstruction = model.predict(
        x, batch_size=512, verbose=0
    )
    return np.abs(reconstruction[:, -1] - x[:, -1, :2])


def reconstruction_scores(data, model, bundle):
    """Die Rekonstruktion nutzt beide Größen; der Alarm bewertet Besucher."""
    errors = reconstruction_errors(model, data)
    target = FEATURES.index(bundle.get("anomaly_feature", ANOMALY_FEATURE))
    scores = errors[:, target] / np.asarray(bundle["error_scale"])[target]
    scores[~data["observed"]] = np.nan
    return scores


def train():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    scaler = fit_scaler(df)
    data = make_windows(df, scaler)
    masks = split_masks(data)
    x = sequences(data)
    y = x[:, :, :2]
    weights = x[:, :, 2]

    model = build_model()
    model.summary()
    history = model.fit(
        add_masking_noise(x[masks["train"]]),
        y[masks["train"]],
        sample_weight=weights[masks["train"]],
        validation_data=(
            x[masks["tune"]], y[masks["tune"]], weights[masks["tune"]]
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )],
    )

    calibration_errors = reconstruction_errors(model, {
        name: values[masks["calibration"]] for name, values in data.items()
    })
    error_scale = np.maximum(
        np.quantile(calibration_errors, 0.75, axis=0), 0.05
    )
    target = FEATURES.index(ANOMALY_FEATURE)
    calibration_scores = calibration_errors[:, target] / error_scale[target]
    rank = min(
        len(calibration_scores),
        int(np.ceil((len(calibration_scores) + 1) * (1 - ALPHA))),
    )

    bundle = {
        "model": "autoencoder.keras",
        "features": FEATURES,
        "anomaly_feature": ANOMALY_FEATURE,
        "window_size": WINDOW_SIZE,
        "architecture": {"lstm_units": LSTM_UNITS},
        "training": {
            "max_epochs": EPOCHS,
            "epochs_run": len(history.history["loss"]),
            "batch_size": BATCH_SIZE,
            "max_mask_hours": MAX_MASK_HOURS,
        },
        "scaler": scaler,
        "error_scale": error_scale.tolist(),
        "calibrator": {
            "alpha": ALPHA,
            "target_feature": ANOMALY_FEATURE,
            "threshold": float(np.sort(calibration_scores)[rank - 1]),
        },
        "splits": {
            "train_end": str(TRAIN_END),
            "tune_end": str(TUNE_END),
            "calibration_end": str(CALIBRATION_END),
        },
        "parameters": model.count_params(),
        "sample_counts": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
    }
    model.save(MODEL_DIR / bundle["model"])
    with open(MODEL_DIR / "model_bundle.json", "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)

    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validierung")
    plt.xlabel("Epoche")
    plt.ylabel("Huber-Fehler")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=150)
    plt.close()

    print(f"Modell gespeichert: {MODEL_DIR / bundle['model']}")
    print(f"Parameter: {model.count_params():,}")
    return bundle


if __name__ == "__main__":
    train()
