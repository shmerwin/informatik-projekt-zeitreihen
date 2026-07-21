"""Prüft den einfachen LSTM mit synthetischen Anomalien und Rauschen.

Die Injektionen liefern vollständige Testlabels für Konfusionsmatrix,
Precision, Recall, F1, ROC-AUC und zeitliche IoU.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from .evaluate import apply_detector
from .train import (
    CALIBRATION_END, DATA_PATH, FEATURES, MODEL_DIR, file_hash, inverse_values,
    load_data, make_windows, scale_values,
)


ANOMALY_TYPES = ["spike", "dip", "level_shift", "drift", "stuck_at"]
SEEDS = [101, 202, 303]


def classification_metrics(labels, alarms, scores):
    labels, alarms = np.asarray(labels, bool), np.asarray(alarms, bool)
    tn, fp, fn, tp = confusion_matrix(
        labels, alarms, labels=[False, True]
    ).ravel()
    union = tp + fp + fn
    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(labels, alarms, zero_division=0)),
        "recall": float(recall_score(labels, alarms, zero_division=0)),
        "f1": float(f1_score(labels, alarms, zero_division=0)),
        "temporal_iou": float(tp / union) if union else 1.0,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def anomaly_scores(data, prediction, bundle):
    error = np.abs(data["y"] - prediction)
    feature_scores = error / np.asarray(bundle["error_scale"])
    limits = np.asarray(bundle["detector"]["feature_thresholds"])
    return np.max(feature_scores / limits, axis=1), feature_scores


def period_data(data, start=CALIBRATION_END, end=None):
    times = pd.to_datetime(data["time"], utc=True)
    mask = times >= start
    if end is not None:
        mask &= times < end
    return {name: values[np.asarray(mask)] for name, values in data.items()}


def select_events(df, anomaly_type, seed, start=CALIBRATION_END, end=None,
                  allowed_keys=None):
    """Wählt feste Ereignisse, ohne Vorhersagen eines Modells zu betrachten."""
    length = 1 if anomaly_type in {"spike", "dip"} else (
        6 if anomaly_type == "stuck_at" else 4
    )
    period = df[(df["timestamp"] >= start) & ~df["is_imputed"]]
    if end is not None:
        period = period[period["timestamp"] < end]
    candidates = []
    for _, group in period.groupby("name"):
        group = group.sort_values("timestamp")
        indices = group.index.to_numpy()
        times = pd.DatetimeIndex(group["timestamp"])
        for offset in range(len(group) - length + 1):
            segment = indices[offset:offset + length]
            segment_times = times[offset:offset + length]
            keys = set(zip(
                group.loc[segment, "name"],
                pd.to_datetime(group.loc[segment, "timestamp"], utc=True),
            ))
            continuous = length == 1 or np.all(
                np.diff(segment_times) == pd.Timedelta(hours=1)
            )
            permitted = allowed_keys is None or keys.issubset(allowed_keys)
            if continuous and permitted:
                candidates.append(segment)

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    target = 20 if length == 1 else 12
    events, used_sensors = [], set()
    for number, segment in enumerate(candidates):
        sensor = df.loc[segment[0], "name"]
        if sensor in used_sensors:
            continue
        direction = 1 if anomaly_type == "spike" else -1
        if anomaly_type in {"level_shift", "drift"}:
            direction = 1 if len(events) % 2 == 0 else -1
        events.append({
            "indices": segment,
            "feature": FEATURES[len(events) % len(FEATURES)],
            "direction": direction,
        })
        used_sensors.add(sensor)
        if len(events) == target:
            break
    return events


def inject(df, events, anomaly_type, strength, scaler):
    """Verändert eine Kopie; die Originaldaten bleiben unangetastet."""
    injected = df.copy()
    labels = set()
    realized_strengths = []
    for event in events:
        feature_id = FEATURES.index(event["feature"])
        original_values = injected.loc[event["indices"], event["feature"]].copy()
        event_labels, event_strengths = set(), []
        first = injected.loc[event["indices"][0]]
        previous = injected[
            (injected["name"] == first["name"])
            & (injected["timestamp"] < first["timestamp"])
            & ~injected["is_imputed"]
        ].sort_values("timestamp").tail(1)
        frozen = (
            float(previous[event["feature"]].iloc[0])
            if not previous.empty else float(first[event["feature"]])
        )

        for position, index in enumerate(event["indices"]):
            row = injected.loc[index]
            event_labels.add((
                row["name"], pd.to_datetime(row["timestamp"], utc=True)
            ))
            scaled = scale_values(row[FEATURES].to_numpy(float)[None, :], scaler)[0]
            original_scaled = float(scaled[feature_id])
            if anomaly_type in {"spike", "dip", "level_shift"}:
                scaled[feature_id] += event["direction"] * strength
                value = inverse_values(scaled[None, :], scaler)[0, feature_id]
            elif anomaly_type == "drift":
                scaled[feature_id] += event["direction"] * strength * (
                    (position + 1) / len(event["indices"])
                )
                value = inverse_values(scaled[None, :], scaler)[0, feature_id]
            else:
                value = frozen
            if event["feature"] == "visitors":
                value = round(value)
            injected.loc[index, event["feature"]] = max(value, 0)
            changed = scale_values(
                injected.loc[index, FEATURES].to_numpy(float)[None, :], scaler
            )[0, feature_id]
            event_strengths.append(abs(float(changed) - original_scaled))

        # Ein nahezu unverändertes Stuck-at-Segment ist keine belastbare Anomalie.
        if anomaly_type == "stuck_at" and np.mean(event_strengths) < 0.5:
            injected.loc[event["indices"], event["feature"]] = original_values
            continue
        labels.update(event_labels)
        realized_strengths.extend(event_strengths)
    return injected, labels, realized_strengths


def event_recall(labels, predictions, sensors, times):
    frame = pd.DataFrame({
        "sensor": sensors, "time": pd.to_datetime(times, utc=True),
        "label": labels, "prediction": predictions,
    })
    detected, total = 0, 0
    for _, group in frame.groupby("sensor"):
        active = group[group["label"]].sort_values("time")
        previous, current = None, []
        events = []
        for index, time in zip(active.index, active["time"]):
            if previous is None or time - previous == pd.Timedelta(hours=1):
                current.append(index)
            else:
                events.append(current)
                current = [index]
            previous = time
        if current:
            events.append(current)
        total += len(events)
        detected += sum(bool(frame.loc[event, "prediction"].any()) for event in events)
    return float(detected / total) if total else 0.0


def noise_robustness(base, model, bundle):
    observed = base["observed"]
    clean_prediction = model.predict(base["X"], verbose=0)
    clean_mae = np.mean(np.abs(
        base["y_raw"][observed]
        - inverse_values(clean_prediction[observed], bundle["scaler"])
    ), axis=0)
    result = {}
    for noise in [0.1, 0.25]:
        rng = np.random.default_rng(SEEDS[0])
        noisy_x = base["X"].copy()
        noisy_x[:, :, :2] += rng.normal(0, noise, noisy_x[:, :, :2].shape)
        prediction = model.predict(noisy_x, verbose=0)
        mae = np.mean(np.abs(
            base["y_raw"][observed]
            - inverse_values(prediction[observed], bundle["scaler"])
        ), axis=0)
        result[str(noise)] = {
            feature: {
                "mae": float(mae[i]),
                "relative_to_clean": float(mae[i] / max(clean_mae[i], 1e-9)),
            }
            for i, feature in enumerate(FEATURES)
        }
    return result


def run_benchmark():
    with open(MODEL_DIR / "model_bundle.json", encoding="utf-8") as file:
        bundle = json.load(file)
    model_path = MODEL_DIR / bundle["model"]
    if file_hash(DATA_PATH) != bundle["data_sha256"]:
        raise ValueError("Datensatz passt nicht zum gespeicherten Modell")
    if file_hash(model_path) != bundle["model_sha256"]:
        raise ValueError("Modelldatei passt nicht zu model_bundle.json")

    df = load_data()
    base = period_data(make_windows(df, bundle["scaler"]))
    model = tf.keras.models.load_model(model_path, compile=False)
    base_prediction = model.predict(base["X"], verbose=0)
    base_score, base_feature_scores = anomaly_scores(
        base, base_prediction, bundle
    )
    base_alarm, _ = apply_detector(
        base_feature_scores, base["observed"], bundle
    )
    keys = [
        (sensor, pd.to_datetime(time, utc=True))
        for sensor, time in zip(base["sensor"], base["time"])
    ]
    rows = []
    curve_data = {name: {"labels": [], "scores": []} for name in ANOMALY_TYPES}
    for seed in SEEDS:
        for anomaly_type in ANOMALY_TYPES:
            events = select_events(df, anomaly_type, seed)
            strengths = [1.0] if anomaly_type == "stuck_at" else [0.5, 1.0, 2.0]
            for strength in strengths:
                injected_df, label_keys, realized = inject(
                    df, events, anomaly_type, strength, bundle["scaler"]
                )
                if not label_keys:
                    continue
                injected = period_data(make_windows(injected_df, bundle["scaler"]))
                injected_keys = [
                    (sensor, pd.to_datetime(time, utc=True))
                    for sensor, time in zip(injected["sensor"], injected["time"])
                ]
                if injected_keys != keys:
                    raise ValueError("Injektion hat die Zeitachse verändert")
                prediction = model.predict(injected["X"], verbose=0)
                score, feature_scores = anomaly_scores(
                    injected, prediction, bundle
                )
                alarm, _ = apply_detector(
                    feature_scores, injected["observed"], bundle
                )
                incremental_alarm = alarm & ~base_alarm
                labels = np.array([key in label_keys for key in keys])
                observed = injected["observed"]
                metrics = classification_metrics(
                    labels[observed], incremental_alarm[observed],
                    (score - base_score)[observed],
                )
                if strength == 2.0 or anomaly_type == "stuck_at":
                    curve_data[anomaly_type]["labels"].append(labels[observed])
                    curve_data[anomaly_type]["scores"].append(
                        (score - base_score)[observed]
                    )
                rows.append({
                    "seed": seed,
                    "anomaly_type": anomaly_type,
                    "strength": strength,
                    "events": len(label_keys) // (
                        1 if anomaly_type in {"spike", "dip"}
                        else 6 if anomaly_type == "stuck_at" else 4
                    ),
                    "mean_realized_strength": float(np.mean(realized)),
                    **{key: value for key, value in metrics.items()
                       if key != "confusion_matrix"},
                    **metrics["confusion_matrix"],
                    "event_recall": event_recall(
                        labels[observed], incremental_alarm[observed],
                        injected["sensor"][observed], injected["time"][observed],
                    ),
                })

    results = pd.DataFrame(rows)
    summary = (results.groupby(["anomaly_type", "strength"])
               .mean(numeric_only=True).reset_index())
    report = {
        "design": (
            "Anomalien werden in feste, ursprünglich gemessene Testsegmente "
            "injiziert, die unabhängig von Modellalarmen gewählt wurden. "
            "Bewertet werden zusätzliche Alarme gegenüber dem Grundlauf."
        ),
        "seeds": SEEDS,
        "detector": {
            **bundle["detector"],
        },
        "metrics": summary.to_dict(orient="records"),
        "noise_robustness": noise_robustness(base, model, bundle),
        "warning": "Synthetische Labels ersetzen keine fachlich bestätigten Ereignisse.",
    }
    summary.to_csv(MODEL_DIR / "injection_benchmark.csv", index=False)
    with open(MODEL_DIR / "injection_benchmark.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)

    plt.figure(figsize=(7, 5))
    for anomaly_type, values in curve_data.items():
        if not values["labels"]:
            continue
        labels = np.concatenate(values["labels"])
        scores = np.concatenate(values["scores"])
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, label=f"{anomaly_type} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Kurven der starken synthetischen Anomalien")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "injection_roc_curve.png", dpi=150)
    plt.close()
    print(f"Benchmark gespeichert: {MODEL_DIR / 'injection_benchmark.csv'}")
    return report


if __name__ == "__main__":
    run_benchmark()
