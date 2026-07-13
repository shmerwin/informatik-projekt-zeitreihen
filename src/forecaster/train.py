''' LSTM-Forecaster: sagt aus den letzten 24 Stunden die nächste Stunde voraus.
Eine Stunde, in der die echte Messung stark von der Vorhersage abweicht,
ist ein Anomalie-Kandidat.
'''

import os
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Reproduzierbarkeit
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed",
                         "cleaned_sensordata.json")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "saved_models", "forecaster")

FEATURES = ["visitors", "avgDuration"]
WINDOW_SIZE = 24

# Diese zwei Standorte besitzen mit 36 bzw. 59 Stunden zu kurze Zeitreihen.
EXCLUDE_SENSORS = [
    "Kurpark - Eingang Ludwigstraße",
    "Friedrichstraße 1",
]

# Trainingsdaten sind Vergangenheit, Val Daten Zukunft
TRAIN_END = pd.Timestamp("2025-07-08", tz="UTC")
VAL_END = pd.Timestamp("2025-07-10", tz="UTC")

EPOCHS = 100
BATCH_SIZE = 128


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} nicht gefunden. "
            "Bitte zuerst src/preprocessing.py ausführen.")

    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["sensordata"])
    if "is_imputed" not in df.columns:
        raise ValueError(
            "is_imputed fehlt. Bitte zuerst src/preprocessing.py ausführen.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[~df["name"].isin(EXCLUDE_SENSORS)]

    return df.sort_values(["name", "timestamp"])


def make_windows(df):
    # Erzeugt je Sensor Fenster von 24 Stunden und 1 Stunde Vorhersage
    data = {split: {"X": [], "y": [], "sensor": [], "time": []}
            for split in ("train", "val", "test")}

    for name, group in df.groupby("name"):
        # je Sensor skalieren
        # gefittet nur auf dem Trainingszeitraum
        scaler = MinMaxScaler()
        scaler.fit(group.loc[group["timestamp"] < TRAIN_END, FEATURES])
        values = scaler.transform(group[FEATURES])
        times = group["timestamp"].to_list()
        imputed = group["is_imputed"].to_numpy(dtype=bool)

        for i in range(WINDOW_SIZE, len(values)):
            # Imputierte Werte dürfen Kontext sein, aber nie Lern-/Prüfziel.
            if imputed[i]:
                continue
            t = times[i]
            split = ("train" if t < TRAIN_END
                     else "val" if t < VAL_END else "test")
            data[split]["X"].append(values[i - WINDOW_SIZE:i])
            data[split]["y"].append(values[i])
            data[split]["sensor"].append(name)
            data[split]["time"].append(t)

    for d in data.values():
        d["X"] = np.array(d["X"], dtype="float32")
        d["y"] = np.array(d["y"], dtype="float32")

    return data


if __name__ == "__main__":
    df = load_data()
    data = make_windows(df)

    print("Train shape:", data["train"]["X"].shape)
    print("Val shape  :", data["val"]["X"].shape)
    print("Test shape :", data["test"]["X"].shape)

    model = models.Sequential([
        layers.Input(shape=(WINDOW_SIZE, len(FEATURES))),
        layers.LSTM(64),
        layers.Dense(len(FEATURES))
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    history = model.fit(
        data["train"]["X"], data["train"]["y"],
        validation_data=(data["val"]["X"], data["val"]["y"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "forecaster.keras"))
    print("Modell gespeichert:", os.path.join(MODEL_DIR, "forecaster.keras"))

    # Trainingsverlauf
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Trainingsverlauf")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)
    plt.show()
