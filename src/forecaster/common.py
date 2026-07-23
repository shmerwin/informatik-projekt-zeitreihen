"""Datenlogik für beide Modelle und Hilfsfunktionen des Forecasters."""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_sensordata.json"
MODEL_DIR = ROOT / "saved_models" / "forecaster"
FEATURES = ["visitors", "avgDuration"]
WINDOW_SIZE = 24
QUANTILES = [0.1, 0.5, 0.9]
TRAIN_END = pd.Timestamp("2025-07-08", tz="UTC")
TUNE_END = pd.Timestamp("2025-07-09", tz="UTC")
CALIBRATION_END = pd.Timestamp("2025-07-10", tz="UTC")
ALPHA = 0.01
SEED = 42


def set_seed(seed=SEED):
    tf.keras.utils.set_random_seed(seed)


def load_data(path=DATA_PATH):
    with open(path, encoding="utf-8") as datei:
        df = pd.DataFrame(json.load(datei)["sensordata"])
    pflicht = {"timestamp", "name", "is_imputed", *FEATURES}
    if not pflicht.issubset(df.columns):
        raise ValueError(f"Fehlende Spalten: {sorted(pflicht - set(df.columns))}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["is_imputed"] = df["is_imputed"].astype(bool)
    if df.duplicated(["name", "timestamp"]).any():
        raise ValueError("Doppelte Sensorstunden nach dem Preprocessing")
    if not np.isfinite(df[FEATURES].to_numpy(float)).all():
        raise ValueError("NaN oder unendliche Modellwerte gefunden")

    # Sehr kurze Reihen liefern nicht einmal drei unabhängige Tageszyklen.
    train = df[(df["timestamp"] < TRAIN_END) & ~df["is_imputed"]]
    genug = train.groupby("name").size()
    sensoren = sorted(genug[genug >= 72].index)
    return df[df["name"].isin(sensoren)].sort_values(
        ["name", "timestamp"]
    ).reset_index(drop=True)


def fit_scaler(df):
    """Robuste Log-Skalierung, nur auf gemessenen Trainingswerten gefittet."""
    sensoren = sorted(df["name"].unique())
    median = np.zeros((len(sensoren), len(FEATURES)))
    scale = np.zeros_like(median)
    train = df[(df["timestamp"] < TRAIN_END) & ~df["is_imputed"]]
    for i, sensor in enumerate(sensoren):
        werte = np.log1p(train.loc[train["name"] == sensor, FEATURES].to_numpy(float))
        median[i] = np.median(werte, axis=0)
        iqr = np.quantile(werte, 0.75, axis=0) - np.quantile(werte, 0.25, axis=0)
        scale[i] = np.maximum(iqr, 0.05)
    return {"sensors": sensoren, "median": median.tolist(), "scale": scale.tolist()}


def scale_values(values, sensor_ids, scaler):
    median = np.asarray(scaler["median"])[np.asarray(sensor_ids).reshape(-1)]
    scale = np.asarray(scaler["scale"])[np.asarray(sensor_ids).reshape(-1)]
    return ((np.log1p(values) - median) / scale).astype("float32")


def inverse_values(values, sensor_ids, scaler):
    median = np.asarray(scaler["median"])[np.asarray(sensor_ids).reshape(-1)]
    scale = np.asarray(scaler["scale"])[np.asarray(sensor_ids).reshape(-1)]
    if np.asarray(values).ndim == 3:
        median, scale = median[:, :, None], scale[:, :, None]
    return np.maximum(np.expm1(np.asarray(values) * scale + median), 0)


def _calendar(times):
    lokal = pd.DatetimeIndex(times).tz_convert("Europe/Berlin")
    stunde = lokal.hour.to_numpy(float)
    wochentag = lokal.dayofweek.to_numpy(float)
    return np.column_stack([
        np.sin(2 * np.pi * stunde / 24), np.cos(2 * np.pi * stunde / 24),
        np.sin(2 * np.pi * wochentag / 7), np.cos(2 * np.pi * wochentag / 7),
        wochentag >= 5,
    ]).astype("float32")


def make_windows(df, scaler):
    """Erzeugt 24h-Fenster samt Kalender, Sensor-ID und Imputationsmaske."""
    daten = {name: [] for name in [
        "history", "future", "sensor_id", "y_scaled", "y_raw", "observed",
        "time", "sensor", "seasonal", "imputed_fraction",
    ]}
    sensor_id = {name: i for i, name in enumerate(scaler["sensors"])}

    for sensor, gruppe in df.groupby("name", sort=False):
        gruppe = gruppe.sort_values("timestamp").reset_index(drop=True)
        sid = sensor_id[sensor]
        roh = gruppe[FEATURES].to_numpy(float)
        skaliert = scale_values(roh, np.full(len(gruppe), sid), scaler)
        beobachtet = ~gruppe["is_imputed"].to_numpy(bool)
        zeiten = pd.DatetimeIndex(gruppe["timestamp"])
        kalender = _calendar(zeiten)
        dynamisch = np.column_stack([skaliert, beobachtet.astype(float), kalender])

        for i in range(WINDOW_SIZE, len(gruppe)):
            ausschnitt = zeiten[i - WINDOW_SIZE:i + 1]
            if not np.all((ausschnitt[1:] - ausschnitt[:-1]) == pd.Timedelta(hours=1)):
                continue
            daten["history"].append(dynamisch[i - WINDOW_SIZE:i])
            daten["future"].append(kalender[i])
            daten["sensor_id"].append([sid])
            daten["y_scaled"].append(skaliert[i])
            daten["y_raw"].append(roh[i])
            daten["observed"].append(beobachtet[i])
            daten["time"].append(zeiten[i].to_datetime64())
            daten["sensor"].append(sensor)
            daten["seasonal"].append(roh[i - 24])
            daten["imputed_fraction"].append(1 - beobachtet[i - WINDOW_SIZE:i].mean())

    arrays = {
        "history": np.asarray(daten["history"], dtype="float32"),
        "future": np.asarray(daten["future"], dtype="float32"),
        "sensor_id": np.asarray(daten["sensor_id"], dtype="int32"),
        "y_scaled": np.asarray(daten["y_scaled"], dtype="float32"),
        "y_raw": np.asarray(daten["y_raw"], dtype=float),
        "observed": np.asarray(daten["observed"], dtype=bool),
        "time": np.asarray(daten["time"], dtype="datetime64[us]"),
        "sensor": np.asarray(daten["sensor"], dtype=object),
        "seasonal": np.asarray(daten["seasonal"], dtype=float),
        "imputed_fraction": np.asarray(daten["imputed_fraction"], dtype=float),
    }
    return arrays


def model_inputs(data, mask=None):
    mask = slice(None) if mask is None else mask
    return {name: data[name][mask] for name in ["history", "future", "sensor_id"]}


def split_masks(data):
    zeit = pd.to_datetime(data["time"], utc=True)
    gemessen = data["observed"]
    return {
        "train": np.asarray((zeit < TRAIN_END) & gemessen),
        "tune": np.asarray((zeit >= TRAIN_END) & (zeit < TUNE_END) & gemessen),
        "calibration": np.asarray(
            (zeit >= TUNE_END) & (zeit < CALIBRATION_END) & gemessen
        ),
        "test": np.asarray((zeit >= CALIBRATION_END) & gemessen),
        "test_all": np.asarray(zeit >= CALIBRATION_END),
    }


@keras.utils.register_keras_serializable(package="forecaster")
class QuantileLoss(keras.losses.Loss):
    """Pinball-Loss für unteres, mittleres und oberes Prognosequantil."""
    def call(self, y_true, y_pred):
        fehler = tf.expand_dims(y_true, -1) - y_pred
        q = tf.constant(QUANTILES, dtype=y_pred.dtype)
        return tf.reduce_mean(tf.maximum(q * fehler, (q - 1) * fehler), axis=(-2, -1))


def build_model(n_sensors, history_channels, params):
    history = keras.Input((WINDOW_SIZE, history_channels), name="history")
    future = keras.Input((5,), name="future")
    sid = keras.Input((1,), dtype="int32", name="sensor_id")
    x = layers.LSTM(params["units"], dropout=params["dropout"])(history)
    sensor = layers.Flatten()(layers.Embedding(n_sensors, params["embedding"])(sid))
    x = layers.Concatenate()([x, sensor, future])
    x = layers.Dense(params["dense"], activation="swish")(x)
    x = layers.Dropout(params["dropout"])(x)
    output = layers.Reshape((len(FEATURES), len(QUANTILES)))(
        layers.Dense(len(FEATURES) * len(QUANTILES))(x)
    )
    model = keras.Model({"history": history, "future": future, "sensor_id": sid}, output)
    model.compile(keras.optimizers.Adam(params["lr"], clipnorm=1), QuantileLoss())
    return model


def predict_quantiles(model, data, mask=None):
    return np.sort(model.predict(
        model_inputs(data, mask), batch_size=512, verbose=0
    ), axis=-1)


def fit_calibrator(actual, prediction, alpha=ALPHA):
    """Kalibriert eine gemeinsame Schwelle für beide Messgrößen."""
    breite = prediction[..., -1] - prediction[..., 0]
    minimum = np.maximum(np.quantile(breite, 0.1, axis=0), 0.05)
    feature_score = np.maximum(
        prediction[..., 0] - actual, actual - prediction[..., -1]
    ) / np.maximum(breite, minimum)
    scores = np.sort(feature_score.max(axis=1))
    rang = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - alpha))))
    return {
        "alpha": alpha, "threshold": float(scores[rang - 1]),
        "minimum_width": minimum.tolist(), "scores": scores.tolist(),
    }


def anomaly_scores(actual, prediction, calibrator):
    minimum = np.asarray(calibrator["minimum_width"])
    breite = np.maximum(prediction[..., -1] - prediction[..., 0], minimum)
    feature = np.maximum(
        prediction[..., 0] - actual, actual - prediction[..., -1]
    ) / breite
    joint = feature.max(axis=1)
    kalibrierung = np.asarray(calibrator["scores"])
    p = (len(kalibrierung) - np.searchsorted(kalibrierung, joint) + 1) / (
        len(kalibrierung) + 1
    )
    return feature, joint, p


def conformal_bounds(prediction, calibrator):
    minimum = np.asarray(calibrator["minimum_width"])
    breite = np.maximum(prediction[..., -1] - prediction[..., 0], minimum)
    abstand = calibrator["threshold"] * breite
    return prediction[..., 0] - abstand, prediction[..., -1] + abstand
