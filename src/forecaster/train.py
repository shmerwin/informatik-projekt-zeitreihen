"""Trainiert und kalibriert den festgelegten LSTM-Forecaster.

Aufruf aus dem Projektordner: ``python -m src.forecaster.train``
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from .common import (
    ALPHA, ANOMALY_FEATURE, DATA_PATH, FEATURES, MODEL_DIR, QUANTILES,
    build_model, evaluation_metadata, fit_calibrator, fit_scaler, load_data,
    make_windows, model_inputs, predict_quantiles, select_by_validation_loss,
    set_seed, split_masks,
)


CANDIDATES = [
    {"units": 16, "dense": 16, "embedding": 2, "dropout": 0.05, "lr": 1e-3},
    {"units": 32, "dense": 32, "embedding": 4, "dropout": 0.15, "lr": 1e-3},
    {"units": 64, "dense": 32, "embedding": 4, "dropout": 0.10, "lr": 1e-3},
    {"units": 64, "dense": 48, "embedding": 6, "dropout": 0.25, "lr": 5e-4},
    {"units": 128, "dense": 64, "embedding": 8, "dropout": 0.30, "lr": 3e-4},
]
MAX_EPOCHS = 80
PATIENCE = 8
BATCH_SIZE = 512
INITIAL_EPOCHS = 10
REDUCTION_FACTOR = 2


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    scaler = fit_scaler(df)
    data = make_windows(df, scaler)
    masks = split_masks(data)
    for split in ["train", "tune", "calibration", "test"]:
        if not masks[split].any():
            raise ValueError(f"Split '{split}' enthält keine gemessenen Ziele")

    train_inputs = model_inputs(data, masks["train"])
    train_targets = data["y_scaled"][masks["train"]]
    tune_data = (
        model_inputs(data, masks["tune"]), data["y_scaled"][masks["tune"]],
    )

    def run_candidate(params, candidate, initial_epoch, target_epoch, stop):
        if candidate is None:
            set_seed()
            candidate = build_model(
                len(scaler["sensors"]), data["history"].shape[-1], params
            )
        history = candidate.fit(
            train_inputs,
            train_targets,
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
    model = build_model(
        len(scaler["sensors"]), data["history"].shape[-1], selected["params"]
    )
    model.summary()
    history = model.fit(
        model_inputs(data, training),
        data["y_scaled"][training],
        epochs=selected["best_epoch"],
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
        "anomaly_feature": ANOMALY_FEATURE,
        "quantiles": QUANTILES,
        "window_size": 24,
        "architecture": selected["params"],
        "epochs": selected["best_epoch"],
        "parameters": model.count_params(),
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
            "sample_count": int(training.sum()),
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": int(
                (training.sum() + BATCH_SIZE - 1) // BATCH_SIZE
            ),
            "optimizer_steps": int(
                selected["best_epoch"]
                * ((training.sum() + BATCH_SIZE - 1) // BATCH_SIZE)
            ),
        },
        "scaler": scaler,
        "calibrator": calibrator,
        **evaluation_metadata(data, masks),
    }
    model.save(MODEL_DIR / bundle["model"])
    with open(MODEL_DIR / "model_bundle.json", "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training")
    plt.xlabel("Epoche")
    plt.ylabel("Quantile-Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=150)
    plt.close()

    print(f"Modell gespeichert: {MODEL_DIR / bundle['model']}")
    print(f"Parameter: {model.count_params():,}")
    print("Alarmregel und Modellvergleich: python -m src.benchmark")
    return bundle


if __name__ == "__main__":
    train()
