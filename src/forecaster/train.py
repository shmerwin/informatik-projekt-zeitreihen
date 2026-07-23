"""Trainiert und kalibriert den festgelegten LSTM-Forecaster.

Aufruf aus dem Projektordner: ``python -m src.forecaster.train``
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from .common import (
    ALPHA, CALIBRATION_END, DATA_PATH, FEATURES, MODEL_DIR, QUANTILES,
    TRAIN_END, TUNE_END, build_model, fit_calibrator, fit_scaler,
    load_data, make_windows, model_inputs, predict_quantiles, set_seed,
    split_masks,
)


# Feste, bereits ausgewählte Architektur. Keine erneute Modell-Suche pro Lauf.
PARAMS = {
    "units": 64,
    "dense": 48,
    "embedding": 6,
    "dropout": 0.25,
    "lr": 5e-4,
}
EPOCHS = 21
BATCH_SIZE = 128


def train(data_path=DATA_PATH, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    scaler = fit_scaler(df)
    data = make_windows(df, scaler)
    masks = split_masks(data)
    for split in ["train", "tune", "calibration", "test"]:
        if not masks[split].any():
            raise ValueError(f"Split '{split}' enthält keine gemessenen Ziele")

    # Tuning wurde nur zur Festlegung der Architektur benutzt und gehört beim
    # endgültigen Lauf zum Training. Kalibrierung und Test bleiben getrennt.
    training = masks["train"] | masks["tune"]
    set_seed()
    model = build_model(
        len(scaler["sensors"]), data["history"].shape[-1], PARAMS
    )
    model.summary()
    history = model.fit(
        model_inputs(data, training),
        data["y_scaled"][training],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    calibration_prediction = predict_quantiles(
        model, data, masks["calibration"]
    )
    calibrator = fit_calibrator(
        data["y_scaled"][masks["calibration"]],
        calibration_prediction,
        ALPHA,
    )

    bundle = {
        "model": "forecaster.keras",
        "features": FEATURES,
        "quantiles": QUANTILES,
        "window_size": 24,
        "architecture": PARAMS,
        "epochs": EPOCHS,
        "parameters": model.count_params(),
        "splits": {
            "train_end": str(TRAIN_END),
            "tune_end": str(TUNE_END),
            "calibration_end": str(CALIBRATION_END),
        },
        "scaler": scaler,
        "calibrator": calibrator,
        "sample_counts": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
    }
    model.save(model_dir / bundle["model"])
    with open(model_dir / "model_bundle.json", "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training")
    plt.xlabel("Epoche")
    plt.ylabel("Quantile-Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_history.png", dpi=150)
    plt.close()

    print(f"Modell gespeichert: {model_dir / bundle['model']}")
    print(f"Parameter: {model.count_params():,}")
    print("Alarmregel und Modellvergleich: python -m src.benchmark")
    return bundle


if __name__ == "__main__":
    train()
