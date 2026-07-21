"""Trainiert einen einfachen LSTM-Forecaster.

Aufruf aus dem Projektordner: ``.venv\\Scripts\\python.exe -m src.forecaster.train``
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_sensordata.json"
MODEL_DIR = ROOT / "saved_models" / "forecaster"
FEATURES = ["visitors", "avgDuration"]
WINDOW_SIZE = 24
LSTM_UNITS = 32
HIDDEN_UNITS = 16
DROPOUT = 0.1
REFIT_EPOCHS = 54
TRAIN_END = pd.Timestamp("2025-07-08", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-07-09", tz="UTC")
CALIBRATION_END = pd.Timestamp("2025-07-10", tz="UTC")
SEED = 42


def file_hash(path):
    """SHA-256 verbindet Modell, Konfiguration und Datensatz eindeutig."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path=DATA_PATH):
    with open(path, encoding="utf-8") as file:
        df = pd.DataFrame(json.load(file)["sensordata"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["is_imputed"] = df["is_imputed"].astype(bool)

    # Mindestens drei gemessene Trainingstage pro Sensor.
    measured_train = df[(df["timestamp"] < TRAIN_END) & ~df["is_imputed"]]
    counts = measured_train.groupby("name").size()
    sensors = counts[counts >= 72].index
    return (df[df["name"].isin(sensors)]
            .sort_values(["name", "timestamp"]).reset_index(drop=True))


def fit_scaler(df):
    """StandardScaler auf log-transformierten, gemessenen Trainingswerten."""
    train = df[(df["timestamp"] < TRAIN_END) & ~df["is_imputed"]]
    scaler = StandardScaler().fit(np.log1p(train[FEATURES].to_numpy(float)))
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def scale_values(values, scaler):
    return ((np.log1p(values) - np.asarray(scaler["mean"]))
            / np.asarray(scaler["scale"])).astype("float32")


def inverse_values(values, scaler):
    restored = np.expm1(
        np.asarray(values) * np.asarray(scaler["scale"])
        + np.asarray(scaler["mean"])
    )
    return np.maximum(restored, 0)


def make_windows(df, scaler):
    """Erzeugt je Sensor einen Stundenkontext und die folgende Zielstunde."""
    data = {name: [] for name in [
        "X", "y", "y_raw", "history_raw", "observed", "time", "sensor",
        "context_imputed_fraction",
    ]}

    for sensor, group in df.groupby("name", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        raw = group[FEATURES].to_numpy(float)
        scaled = scale_values(raw, scaler)
        observed = ~group["is_imputed"].to_numpy(bool)
        times = pd.DatetimeIndex(group["timestamp"])
        model_values = np.column_stack([scaled, observed.astype("float32")])

        for i in range(WINDOW_SIZE, len(group)):
            section = times[i - WINDOW_SIZE:i + 1]
            if not np.all(np.diff(section) == pd.Timedelta(hours=1)):
                continue
            data["X"].append(model_values[i - WINDOW_SIZE:i])
            data["y"].append(scaled[i])
            data["y_raw"].append(raw[i])
            data["history_raw"].append(raw[i - WINDOW_SIZE:i])
            data["observed"].append(observed[i])
            data["time"].append(times[i].to_datetime64())
            data["sensor"].append(sensor)
            data["context_imputed_fraction"].append(
                1 - observed[i - WINDOW_SIZE:i].mean()
            )

    return {
        "X": np.asarray(data["X"], dtype="float32"),
        "y": np.asarray(data["y"], dtype="float32"),
        "y_raw": np.asarray(data["y_raw"], dtype=float),
        "history_raw": np.asarray(data["history_raw"], dtype=float),
        "observed": np.asarray(data["observed"], dtype=bool),
        "time": np.asarray(data["time"], dtype="datetime64[us]"),
        "sensor": np.asarray(data["sensor"], dtype=object),
        "context_imputed_fraction": np.asarray(
            data["context_imputed_fraction"], dtype=float
        ),
    }


def split_masks(data):
    times = pd.to_datetime(data["time"], utc=True)
    measured = data["observed"]
    return {
        "train": np.asarray((times < TRAIN_END) & measured),
        "validation": np.asarray(
            (times >= TRAIN_END) & (times < VALIDATION_END) & measured
        ),
        "calibration": np.asarray(
            (times >= VALIDATION_END) & (times < CALIBRATION_END) & measured
        ),
        "test": np.asarray((times >= CALIBRATION_END) & measured),
        "test_all": np.asarray(times >= CALIBRATION_END),
    }


def build_model():
    model = models.Sequential([
        layers.Input(shape=(WINDOW_SIZE, len(FEATURES) + 1)),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(HIDDEN_UNITS, activation="relu"),
        layers.Dense(len(FEATURES)),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train(epochs=REFIT_EPOCHS):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    df = load_data()
    scaler = fit_scaler(df)
    data = make_windows(df, scaler)
    masks = split_masks(data)
    # Derselbe feste Trainingsumfang wird auch im Baselinevergleich verwendet.
    model = build_model()
    refit = masks["train"] | masks["validation"]
    history = model.fit(
        data["X"][refit], data["y"][refit],
        epochs=epochs, batch_size=128, verbose=2,
    )

    calibration_prediction = model.predict(
        data["X"][masks["calibration"]], verbose=0
    )
    calibration_error = np.abs(
        data["y"][masks["calibration"]] - calibration_prediction
    )
    error_scale = np.maximum(np.median(calibration_error, axis=0), 0.05)
    feature_scores = calibration_error / error_scale
    feature_thresholds = np.quantile(
        feature_scores, 0.99, axis=0, method="higher"
    )

    model_path = MODEL_DIR / "forecaster.keras"
    model.save(model_path)
    bundle = {
        "version": "simple_sequential_lstm_v3",
        "model": model_path.name,
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "input_channels": [*FEATURES, "is_observed"],
        "splits": {
            "train_end": str(TRAIN_END),
            "validation_end": str(VALIDATION_END),
            "calibration_end": str(CALIBRATION_END),
        },
        "scaler": scaler,
        "error_scale": error_scale.tolist(),
        "detector": {
            "type": "feature_q99",
            "feature_thresholds": feature_thresholds.tolist(),
        },
        "architecture": {
            "lstm_units": LSTM_UNITS,
            "dropout": DROPOUT,
            "hidden_dense_units": HIDDEN_UNITS,
            "loss": "mse",
            "parameters": model.count_params(),
        },
        "training": {
            "seed": SEED,
            "epochs": epochs,
            "batch_size": 128,
        },
        "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "data_sha256": file_hash(DATA_PATH),
        "model_sha256": file_hash(model_path),
        "software": {
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
        },
    }
    with open(MODEL_DIR / "model_bundle.json", "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training")
    plt.xlabel("Epoche")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=150)
    plt.close()

    print(f"LSTM({LSTM_UNITS}) für {epochs} Epochen trainiert")
    print("Merkmalsschwellen: " + ", ".join(
        f"{feature}={value:.3f}"
        for feature, value in zip(FEATURES, feature_thresholds)
    ))
    print(f"Modellparameter: {model.count_params()}")
    return bundle


if __name__ == "__main__":
    train()
