"""Trainiert und kalibriert den LSTM-Forecaster.

Aufruf aus dem Projektordner: ``python -m src.forecaster.train``
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .common import (
    ALPHA, CALIBRATION_END, DATA_PATH, FEATURES, MODEL_DIR, QUANTILES,
    SEED, TRAIN_END, TUNE_END, build_model, fit_calibrator,
    fit_scaler, inverse_values, load_data, make_windows, model_inputs,
    predict_quantiles, set_seed, split_masks,
)
from .benchmark import select_detector


# Zwei kleine Konfigurationen reichen für diesen sehr kurzen Datensatz.
# Eine große Suche würde die 24 Tuningstunden nur überanpassen.
KANDIDATEN = [
    {"name": "small", "units": 32, "dense": 32, "embedding": 4,
     "dropout": 0.10, "lr": 1e-3},
    {"name": "regularized", "units": 64, "dense": 48, "embedding": 6,
     "dropout": 0.25, "lr": 5e-4},
]


def _mae_je_feature(ist, prognose):
    return {
        feature: float(np.mean(np.abs(ist[:, i] - prognose[:, i])))
        for i, feature in enumerate(FEATURES)
    }


def train(data_path=DATA_PATH, model_dir=MODEL_DIR, max_epochs=100):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    scaler = fit_scaler(df)
    daten = make_windows(df, scaler)
    masken = split_masks(daten)
    for split in ["train", "tune", "calibration", "test"]:
        if not masken[split].any():
            raise ValueError(f"Split '{split}' enthält keine gemessenen Ziele")

    tune_ist = daten["y_raw"][masken["tune"]]
    baseline_mae = _mae_je_feature(
        tune_ist, daten["seasonal"][masken["tune"]]
    )
    berichte = []
    verlaufe = {}

    # Auswahl ausschließlich auf dem Tuning-Tag.
    for nummer, parameter in enumerate(KANDIDATEN):
        tf.keras.backend.clear_session()
        set_seed(SEED + nummer)
        modell = build_model(
            len(scaler["sensors"]), daten["history"].shape[-1], parameter
        )
        verlauf = modell.fit(
            model_inputs(daten, masken["train"]),
            daten["y_scaled"][masken["train"]],
            validation_data=(
                model_inputs(daten, masken["tune"]),
                daten["y_scaled"][masken["tune"]],
            ),
            epochs=max_epochs,
            batch_size=128,
            verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=10, min_delta=1e-4, restore_best_weights=True
            )],
        )
        quantile = predict_quantiles(modell, daten, masken["tune"])
        q50 = inverse_values(quantile[..., 1], daten["sensor_id"][masken["tune"]], scaler)
        mae = _mae_je_feature(tune_ist, q50)
        vergleich = float(np.mean([
            mae[feature] / max(baseline_mae[feature], 1e-9)
            for feature in FEATURES
        ]))
        beste_epoche = int(np.argmin(verlauf.history["val_loss"])) + 1
        berichte.append({
            "parameter": parameter,
            "best_epoch": beste_epoche,
            "mae": mae,
            "mean_mae_relative_to_seasonal": vergleich,
        })
        verlaufe[parameter["name"]] = verlauf.history["val_loss"]

    bestes = min(berichte, key=lambda x: x["mean_mae_relative_to_seasonal"])

    # Danach erneutes Training auf Train + Tuning. Der Kalibrierungstag bleibt
    # unangetastet und bestimmt ausschließlich die Anomalieschwelle.
    refit = masken["train"] | masken["tune"]
    tf.keras.backend.clear_session()
    set_seed()
    modell = build_model(
        len(scaler["sensors"]), daten["history"].shape[-1], bestes["parameter"]
    )
    modell.fit(
        model_inputs(daten, refit), daten["y_scaled"][refit],
        epochs=bestes["best_epoch"], batch_size=128, verbose=2,
    )

    cal_prognose = predict_quantiles(modell, daten, masken["calibration"])
    kalibrierung = fit_calibrator(
        daten["y_scaled"][masken["calibration"]], cal_prognose, ALPHA
    )

    bundle = {
        "model": "forecaster.keras",
        "features": FEATURES,
        "quantiles": QUANTILES,
        "window_size": 24,
        "splits": {
            "train_end": str(TRAIN_END), "tune_end": str(TUNE_END),
            "calibration_end": str(CALIBRATION_END),
        },
        "scaler": scaler,
        "calibrator": kalibrierung,
        "selected": bestes,
        "selection": {
            "seasonal_mae": baseline_mae,
            "candidates": berichte,
        },
        "sample_counts": {name: int(mask.sum()) for name, mask in masken.items()},
    }
    # Die Detektorregel wird mit künstlichen Ereignissen ausschließlich im
    # Kalibrierungsabschnitt gewählt. Der Testabschnitt bleibt hierfür unberührt.
    detektor, detektor_auswahl = select_detector(df, daten, modell, bundle)
    bundle["detector"] = detektor
    bundle["detector_selection"] = detektor_auswahl

    modell.save(model_dir / "forecaster.keras")
    with open(model_dir / "model_bundle.json", "w", encoding="utf-8") as datei:
        json.dump(bundle, datei, ensure_ascii=False, indent=2, allow_nan=False)

    plt.figure(figsize=(8, 4))
    for name, werte in verlaufe.items():
        plt.plot(werte, label=name)
    plt.xlabel("Epoche")
    plt.ylabel("Validation Quantile-Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_history.png", dpi=150)
    plt.close()

    print(f"Gewählt: {bestes['parameter']['name']} "
          f"(Epoche {bestes['best_epoch']})")
    print(f"Gemeinsame Conformal-Schwelle: {kalibrierung['threshold']:.3f}")
    print(f"Gewählte Detektorschwelle: {detektor['point_threshold']:.3f}")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()
    train(args.data, args.model_dir, args.max_epochs)


if __name__ == "__main__":
    main()
