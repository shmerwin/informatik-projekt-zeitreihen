"""Trainiert einen kleinen Transformer für die nächste Sensorstunde.

Aufruf aus dem Projektordner: ``python -m src.transformer.train``
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..forecaster.common import (
    ALPHA, ANOMALY_FEATURE, DATA_PATH, FEATURES, QUANTILES, ROOT, WINDOW_SIZE,
    QuantileLoss, evaluation_metadata, fit_calibrator, fit_scaler, load_data,
    make_windows, model_inputs, predict_quantiles, select_by_validation_loss,
    set_seed, split_masks,
)


MODEL_DIR = ROOT / "saved_models" / "transformer"
CANDIDATES = [
    {"d_model": 16, "heads": 2, "blocks": 1, "ff_dim": 32,
     "dense": 16, "embedding": 2, "dropout": 0.05, "lr": 1e-3},
    {"d_model": 24, "heads": 4, "blocks": 1, "ff_dim": 48,
     "dense": 24, "embedding": 4, "dropout": 0.10, "lr": 1e-3},
    {"d_model": 32, "heads": 4, "blocks": 2, "ff_dim": 64,
     "dense": 32, "embedding": 4, "dropout": 0.10, "lr": 1e-3},
    {"d_model": 48, "heads": 4, "blocks": 2, "ff_dim": 192,
     "dense": 48, "embedding": 8, "dropout": 0.15, "lr": 5e-4},
    {"d_model": 64, "heads": 8, "blocks": 2, "ff_dim": 256,
     "dense": 64, "embedding": 8, "dropout": 0.20, "lr": 3e-4},
    {"d_model": 64, "heads": 4, "blocks": 3, "ff_dim": 256,
     "dense": 64, "embedding": 8, "dropout": 0.30, "lr": 2e-4},
]
MAX_EPOCHS = 120
PATIENCE = 12
BATCH_SIZE = 512
INITIAL_EPOCHS = 10
REDUCTION_FACTOR = 2


def _positions(length, width):
    """Feste Sinus-/Kosinuspositionen aus dem ursprünglichen Transformer."""
    position = np.arange(length)[:, None]
    dimension = np.arange(width)[None, :]
    angle = position / np.power(10000, 2 * (dimension // 2) / width)
    return np.where(dimension % 2 == 0, np.sin(angle), np.cos(angle)).astype(
        "float32"
    )


def _encoder_block(x, params, number):
    """Pre-LayerNorm hält den kleinen Encoder stabil und verständlich."""
    normalized = layers.LayerNormalization(
        epsilon=1e-6, name=f"attention_norm_{number}"
    )(x)
    attention = layers.MultiHeadAttention(
        num_heads=params["heads"],
        key_dim=params["d_model"] // params["heads"],
        dropout=params["dropout"],
        name=f"attention_{number}",
    )(normalized, normalized)
    x = x + layers.Dropout(params["dropout"])(attention)

    normalized = layers.LayerNormalization(
        epsilon=1e-6, name=f"feedforward_norm_{number}"
    )(x)
    feedforward = layers.Dense(
        params["ff_dim"], activation="gelu", name=f"feedforward_{number}"
    )(normalized)
    feedforward = layers.Dropout(params["dropout"])(feedforward)
    feedforward = layers.Dense(params["d_model"])(feedforward)
    return x + layers.Dropout(params["dropout"])(feedforward)


def build_model(n_sensors, history_channels, params):
    """Transformer-Encoder mit Sensor-Kontext und linearem Prognosepfad."""
    if params["d_model"] % params["heads"]:
        raise ValueError("d_model muss durch heads teilbar sein")

    history = keras.Input((WINDOW_SIZE, history_channels), name="history")
    future = keras.Input((5,), name="future")
    sensor_id = keras.Input((1,), dtype="int32", name="sensor_id")

    x = layers.Dense(params["d_model"], name="input_projection")(history)
    x = x + keras.ops.convert_to_tensor(
        _positions(WINDOW_SIZE, params["d_model"])
    )
    sensor = layers.Flatten()(
        layers.Embedding(n_sensors, params["embedding"], name="sensor_embedding")(
            sensor_id
        )
    )
    sensor_context = layers.Dense(
        params["d_model"], use_bias=False, name="sensor_projection"
    )(sensor)
    x = x + layers.RepeatVector(WINDOW_SIZE)(sensor_context)
    x = layers.Dropout(params["dropout"])(x)

    for number in range(params["blocks"]):
        x = _encoder_block(x, params, number)
    x = layers.LayerNormalization(epsilon=1e-6, name="output_norm")(x)

    context = layers.Concatenate()([x[:, -1, :], sensor, future])
    context = layers.Dense(params["dense"], activation="gelu")(context)
    context = layers.Dropout(params["dropout"])(context)
    raw = layers.Reshape((len(FEATURES), len(QUANTILES)))(
        layers.Dense(len(FEATURES) * len(QUANTILES))(context)
    )

    # Ein linearer Pfad bewahrt einfache autoregressive Zusammenhänge.
    measured_history = keras.ops.take(history, [0, 1], axis=-1)
    linear = layers.Dense(len(FEATURES), use_bias=False, name="linear_skip")(
        layers.Flatten()(measured_history)
    )
    median = raw[..., 0] + linear
    lower = median - keras.ops.softplus(raw[..., 1])
    upper = median + keras.ops.softplus(raw[..., 2])
    output = keras.ops.stack([lower, median, upper], axis=-1)

    model = keras.Model(
        {"history": history, "future": future, "sensor_id": sensor_id}, output
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            params["lr"], weight_decay=1e-4, clipnorm=1
        ),
        loss=QuantileLoss(),
    )
    return model


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
            callbacks=[] if not stop else [keras.callbacks.EarlyStopping(
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
        data["y_scaled"][masks["calibration"]], calibration_prediction, ALPHA
    )
    bundle = {
        "model": "transformer.keras",
        "features": FEATURES,
        "anomaly_feature": ANOMALY_FEATURE,
        "quantiles": QUANTILES,
        "window_size": WINDOW_SIZE,
        "architecture": {
            "type": "pre_norm_transformer_with_linear_skip",
            **selected["params"],
        },
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
