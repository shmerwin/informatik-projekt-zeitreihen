"""Trainiert einen kleinen LSTM-Autoencoder für 24-Stunden-Fenster.

Aufruf aus dem Projektordner: ``python -m src.autoencoder.train``
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from ..forecaster.common import (
    ALPHA, ANOMALY_FEATURE, DATA_PATH, FEATURES, ROOT, WINDOW_SIZE,
    add_masking_noise, evaluation_metadata, fit_scaler, load_data,
    make_windows, select_by_validation_loss, set_seed,
    split_masks,
)


MODEL_DIR = ROOT / "saved_models" / "autoencoder"
CANDIDATES = [
    {"lstm_units": 8, "lr": 1e-3, "max_mask_hours": 2},
    {"lstm_units": 16, "lr": 1e-3, "max_mask_hours": 4},
    {"lstm_units": 32, "lr": 5e-4, "max_mask_hours": 4},
    {"lstm_units": 64, "lr": 3e-4, "max_mask_hours": 6},
    {"lstm_units": 16, "lr": 3e-4, "max_mask_hours": 8},
]
MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 512
INITIAL_EPOCHS = 10
REDUCTION_FACTOR = 2


def sequences(data):
    """Verbindet 24 Kontextstunden mit der zu rekonstruierenden Zielstunde."""
    current = np.concatenate([
        data["y_scaled"][:, None, :],
        data["observed"][:, None, None].astype("float32"),
    ], axis=2)
    return np.concatenate([data["history"][:, 1:, :3], current], axis=1)


def build_model(params):
    """Bewusst kompakter Encoder-Decoder nach dem klassischen LSTM-AE-Prinzip."""
    model = models.Sequential([
        layers.Input((WINDOW_SIZE, 3)),
        layers.LSTM(params["lstm_units"]),
        layers.RepeatVector(WINDOW_SIZE),
        layers.LSTM(params["lstm_units"], return_sequences=True),
        layers.Dense(len(FEATURES)),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["lr"]),
        loss=tf.keras.losses.Huber(),
    )
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    scaler = fit_scaler(df)
    data = make_windows(df, scaler)
    masks = split_masks(data)
    x = sequences(data)
    y = x[:, :, :2]
    weights = x[:, :, 2]

    train_x = x[masks["train"]]
    train_y = y[masks["train"]]
    train_weights = weights[masks["train"]]
    tune_data = (
        x[masks["tune"]], y[masks["tune"]], weights[masks["tune"]],
    )

    def run_candidate(params, candidate, initial_epoch, target_epoch, stop):
        if candidate is None:
            set_seed()
            candidate = build_model(params)
        history = candidate.fit(
            add_masking_noise(train_x, params["max_mask_hours"]),
            train_y,
            sample_weight=train_weights,
            validation_data=tune_data,
            initial_epoch=initial_epoch,
            epochs=target_epoch,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[] if not stop else [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, min_delta=1e-4,
                restore_best_weights=True,
            )],
        )
        return history, candidate, candidate.count_params()

    tf.keras.backend.clear_session()
    selected, trials = select_by_validation_loss(
        CANDIDATES, run_candidate, INITIAL_EPOCHS, MAX_EPOCHS,
        REDUCTION_FACTOR,
    )
    training = masks["train"] | masks["tune"]
    tf.keras.backend.clear_session()
    set_seed()
    model = build_model(selected["params"])
    model.summary()
    history = model.fit(
        add_masking_noise(
            x[training], selected["params"]["max_mask_hours"]
        ),
        y[training],
        sample_weight=weights[training],
        epochs=selected["best_epoch"],
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    loss_history = [float(value) for value in history.history["loss"]]

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
        "architecture": selected["params"],
        "epochs": selected["best_epoch"],
        "model_search": {
            "objective": "successive halving; minimum tune val_loss",
            "initial_epochs": INITIAL_EPOCHS,
            "reduction_factor": REDUCTION_FACTOR,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "trials": trials,
            "selected": selected,
        },
        "training": {
            "max_epochs": MAX_EPOCHS,
            "epochs_run": len(loss_history),
            "sample_count": int(training.sum()),
            "batch_size": BATCH_SIZE,
            "loss_history": loss_history,
            "max_mask_hours": selected["params"]["max_mask_hours"],
            "steps_per_epoch": int(
                (training.sum() + BATCH_SIZE - 1) // BATCH_SIZE
            ),
            "optimizer_steps": int(
                selected["best_epoch"]
                * ((training.sum() + BATCH_SIZE - 1) // BATCH_SIZE)
            ),
        },
        "scaler": scaler,
        "error_scale": error_scale.tolist(),
        "calibrator": {
            "alpha": ALPHA,
            "target_feature": ANOMALY_FEATURE,
            "threshold": float(np.sort(calibration_scores)[rank - 1]),
        },
        "parameters": model.count_params(),
        **evaluation_metadata(data, masks),
    }
    model.save(MODEL_DIR / bundle["model"])
    with open(MODEL_DIR / "model_bundle.json", "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)

    search_history = selected["history"]
    epochs = range(1, len(search_history["loss"]) + 1)
    plt.figure(figsize=(6, 3))
    plt.plot(epochs, search_history["loss"], label="Training")
    plt.plot(epochs, search_history["val_loss"], label="Tuning")
    plt.axvline(
        selected["best_epoch"], color="black", linestyle="--", linewidth=1,
        label=f"Beste Epoche: {selected['best_epoch']}",
    )
    plt.xlim(left=1)
    plt.xlabel("Epoche", fontsize=14)
    plt.ylabel("Huber-Fehler", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=150)
    plt.close()

    print(f"Modell gespeichert: {MODEL_DIR / bundle['model']}")
    print(f"Parameter: {model.count_params():,}")
    return bundle


if __name__ == "__main__":
    train()
