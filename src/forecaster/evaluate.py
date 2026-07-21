"""Evaluiert den einfachen LSTM und drei nachvollziehbare Baselines.

Aufruf: ``.venv\\Scripts\\python.exe -m src.forecaster.evaluate``
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from .train import (
    DATA_PATH, FEATURES, MODEL_DIR, TRAIN_END, file_hash,
    inverse_values, load_data, make_windows, split_masks,
)


def regression_metrics(actual, prediction, mase_scale):
    result = {}
    for i, feature in enumerate(FEATURES):
        error = actual[:, i] - prediction[:, i]
        mae = np.mean(np.abs(error))
        result[feature] = {
            "mae": float(mae),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mase": float(mae / max(mase_scale[i], 1e-9)),
        }
    return result


def apply_detector(feature_scores, observed, bundle):
    """Alarmiert, wenn mindestens ein Merkmal seine eigene Grenze übersteigt."""
    limits = np.asarray(bundle["detector"]["feature_thresholds"])
    feature_alarm = np.asarray(feature_scores) > limits
    alarm = np.asarray(observed, bool) & np.any(feature_alarm, axis=1)
    return alarm, feature_alarm


def hour_mean_baseline(df, sensors, times):
    """Sensor- und Stundenmittel ausschließlich aus gemessenen Trainingsdaten."""
    train = df[(df["timestamp"] < TRAIN_END) & ~df["is_imputed"]].copy()
    train["hour"] = train["timestamp"].dt.tz_convert("Europe/Berlin").dt.hour
    profile = train.groupby(["name", "hour"])[FEATURES].mean()
    sensor_mean = train.groupby("name")[FEATURES].mean()

    predictions = []
    local_hours = pd.to_datetime(times, utc=True).tz_convert("Europe/Berlin").hour
    for sensor, hour in zip(sensors, local_hours):
        key = (sensor, int(hour))
        predictions.append(
            profile.loc[key].to_numpy(float)
            if key in profile.index else sensor_mean.loc[sensor].to_numpy(float)
        )
    return np.asarray(predictions)


def event_ids(frame, alarm_column):
    ids = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    number = 0
    for _, group in frame.groupby("sensor", sort=False):
        previous = None
        for index, time in zip(
            group.index[group[alarm_column]],
            group.loc[group[alarm_column], "time"],
        ):
            if previous is None or time - previous != pd.Timedelta(hours=1):
                number += 1
            ids.loc[index] = number
            previous = time
    return ids


def event_table(candidates, id_column, alert_type):
    table = candidates.dropna(subset=[id_column]).groupby(id_column).agg(
        sensor=("sensor", "first"),
        start=("time", "min"),
        end=("time", "max"),
        duration_hours=("time", "size"),
        max_score=("anomaly_score", "max"),
        max_context_imputed_fraction=("context_imputed_fraction", "max"),
    ).reset_index(drop=True)
    table.insert(0, "event_id", [
        f"{alert_type}-{number + 1}" for number in range(len(table))
    ])
    table.insert(1, "alert_type", alert_type)
    return table


def evaluate():
    with open(MODEL_DIR / "model_bundle.json", encoding="utf-8") as file:
        bundle = json.load(file)
    model_path = MODEL_DIR / bundle["model"]
    if file_hash(DATA_PATH) != bundle["data_sha256"]:
        raise ValueError("Datensatz passt nicht zum gespeicherten Modell")
    if file_hash(model_path) != bundle["model_sha256"]:
        raise ValueError("Modelldatei passt nicht zu model_bundle.json")

    df = load_data()
    data = make_windows(df, bundle["scaler"])
    masks = split_masks(data)
    test_mask = masks["test_all"]
    observed = data["observed"][test_mask]
    model = tf.keras.models.load_model(model_path, compile=False)
    prediction_scaled = model.predict(data["X"][test_mask], verbose=0)
    prediction = inverse_values(prediction_scaled, bundle["scaler"])
    scaled_error = np.abs(data["y"][test_mask] - prediction_scaled)
    feature_score = scaled_error / np.asarray(bundle["error_scale"])
    limits = np.asarray(bundle["detector"]["feature_thresholds"])
    score = np.max(feature_score / limits, axis=1)
    forecast_alarm, feature_alarm = apply_detector(
        feature_score, observed, bundle
    )
    missing_alarm = ~observed

    actual = data["y_raw"][test_mask]
    candidates = pd.DataFrame({
        "sensor": data["sensor"][test_mask],
        "time": pd.to_datetime(data["time"][test_mask], utc=True),
        "context_imputed_fraction": data["context_imputed_fraction"][test_mask],
        "anomaly_score": np.where(observed, score, np.nan),
        "forecast_anomaly": forecast_alarm,
        "missing_measurement": missing_alarm,
    })
    for i, feature in enumerate(FEATURES):
        candidates[f"actual_{feature}"] = np.where(observed, actual[:, i], np.nan)
        candidates[f"forecast_{feature}"] = prediction[:, i]
        candidates[f"score_{feature}"] = np.where(
            observed, feature_score[:, i], np.nan
        )
        candidates[f"anomaly_{feature}"] = observed & feature_alarm[:, i]

    candidates["forecast_event_id"] = event_ids(candidates, "forecast_anomaly")
    candidates["missing_event_id"] = event_ids(candidates, "missing_measurement")

    measured_actual = actual[observed]
    train_scale = np.mean(np.abs(
        data["y_raw"][masks["train"]]
        - data["history_raw"][masks["train"], 0]
    ), axis=0)
    baselines = {
        "lstm": prediction[observed],
        "t_minus_24": data["history_raw"][test_mask][observed, 0],
        "hour_mean": hour_mean_baseline(
            df, data["sensor"][test_mask][observed],
            data["time"][test_mask][observed],
        ),
        "rolling_median_24h": np.median(
            data["history_raw"][test_mask][observed], axis=1
        ),
    }
    forecast_metrics = {
        name: regression_metrics(measured_actual, values, train_scale)
        for name, values in baselines.items()
    }

    metrics = {
        "model_version": bundle["version"],
        "forecast": forecast_metrics,
        "counts": {
            "measured_test_points": int(observed.sum()),
            "forecast_candidates": int(forecast_alarm.sum()),
            "forecast_events": int(candidates["forecast_event_id"].nunique()),
            "missing_measurements": int(missing_alarm.sum()),
            "missing_events": int(candidates["missing_event_id"].nunique()),
        },
        "limitation": (
            "Forecast-Treffer sind ohne Vollannotation nur Kandidaten. "
            "Klassifikationsmetriken stehen im Injektionsbenchmark."
        ),
    }

    candidates.to_csv(MODEL_DIR / "anomalies.csv", index=False)
    measured_candidates = candidates[observed]
    measured_candidates[["sensor", "time", "forecast_anomaly", "anomaly_score"]].assign(
        label="", comment=""
    ).to_csv(MODEL_DIR / "labels_template.csv", index=False)
    pd.concat([
        event_table(candidates, "forecast_event_id", "forecast"),
        event_table(candidates, "missing_event_id", "missing"),
    ], ignore_index=True).to_csv(MODEL_DIR / "anomaly_events.csv", index=False)
    with open(MODEL_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=False)

    plot = candidates[candidates["actual_visitors"].notna()]
    sensor = plot.groupby("sensor")["forecast_anomaly"].sum().idxmax()
    plot = plot[plot["sensor"] == sensor]
    plt.figure(figsize=(12, 4))
    plt.plot(plot["time"], plot["actual_visitors"], label="Messung")
    plt.plot(plot["time"], plot["forecast_visitors"], label="LSTM")
    hits = plot[plot["forecast_anomaly"]]
    plt.scatter(hits["time"], hits["actual_visitors"], color="red",
                label="Kandidat", zorder=3)
    plt.title(sensor)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "anomalies_example.png", dpi=150)
    plt.close()

    print(f"Forecast-Kandidaten: {int(forecast_alarm.sum())} Punkte in "
          f"{int(candidates['forecast_event_id'].nunique())} Ereignissen")
    print(f"Fehlende Messungen: {int(missing_alarm.sum())}")
    print("Baselines: t-24, Stundenmittel und gleitender 24h-Median")
    return metrics


if __name__ == "__main__":
    evaluate()
