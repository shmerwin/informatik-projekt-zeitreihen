"""Auswahl und Anwendung der synthetischen Anomalien."""

import numpy as np
import pandas as pd

from .forecaster.common import (
    ANOMALY_FEATURE, FEATURES, inverse_values, scale_values,
)


ANOMALY_TYPES = ["spike", "dip", "level_shift", "drift", "stuck_at"]


def _injection_events(df, anomaly_type, start, end=None, seed=42):
    """Wählt reproduzierbare, ursprünglich gemessene Segmente."""
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
            if length == 1 or np.all(
                (segment_times[1:] - segment_times[:-1]) == pd.Timedelta(hours=1)
            ):
                candidates.append(segment)

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    target_count = 20 if length == 1 else 12
    used_sensors = set()
    events = []
    for segment in candidates:
        sensor = df.loc[segment[0], "name"]
        if sensor in used_sensors:
            continue
        number = len(events)
        direction = 1 if anomaly_type == "spike" or (number // 2) % 2 == 0 else -1
        if anomaly_type == "dip":
            direction = -1
        events.append({
            "indices": segment,
            "feature": ANOMALY_FEATURE,
            "direction": direction,
        })
        used_sensors.add(sensor)
        if len(events) == target_count:
            break
    return events


def _inject(df, events, anomaly_type, strength, scaler):
    """Injiziert vor dem Windowing in eine Kopie der Daten."""
    injected = df.copy()
    sensor_ids = {name: i for i, name in enumerate(scaler["sensors"])}
    labels = set()
    realized_strengths = []

    for event in events:
        feature_id = FEATURES.index(event["feature"])
        original_values = injected.loc[event["indices"], event["feature"]].copy()
        event_labels = set()
        event_strengths = []
        first = injected.loc[event["indices"][0]]
        previous = injected[
            (injected["name"] == first["name"])
            & (injected["timestamp"] < first["timestamp"])
            & ~injected["is_imputed"]
        ].sort_values("timestamp").tail(1)
        frozen_value = (
            float(previous[event["feature"]].iloc[0])
            if not previous.empty else float(first[event["feature"]])
        )

        for position, index in enumerate(event["indices"]):
            row = injected.loc[index]
            original_value = float(row[event["feature"]])
            sensor_id = sensor_ids[row["name"]]
            scaled = scale_values(
                row[FEATURES].to_numpy(float)[None, :],
                np.array([sensor_id]), scaler,
            )[0]
            original = float(scaled[feature_id])

            if anomaly_type in {"spike", "dip", "level_shift"}:
                scaled[feature_id] += event["direction"] * strength
            elif anomaly_type == "drift":
                scaled[feature_id] += event["direction"] * strength * (
                    (position + 1) / len(event["indices"])
                )
            elif anomaly_type == "stuck_at":
                values = row[FEATURES].to_numpy(float)
                values[feature_id] = frozen_value
                scaled = scale_values(
                    values[None, :], np.array([sensor_id]), scaler
                )[0]

            restored = inverse_values(
                scaled[None, :], np.array([sensor_id]), scaler
            )[0]
            value = restored[feature_id]
            if event["feature"] == "visitors":
                value = round(value)
            injected.loc[index, event["feature"]] = value
            actual_scaled = scale_values(
                injected.loc[index, FEATURES].to_numpy(float)[None, :],
                np.array([sensor_id]), scaler,
            )[0, feature_id]
            if float(value) != original_value:
                event_labels.add((
                    row["name"], row["timestamp"].to_datetime64()
                ))
                event_strengths.append(abs(float(actual_scaled) - original))

        if not event_strengths or (
            anomaly_type == "stuck_at" and np.mean(event_strengths) < 0.5
        ):
            injected.loc[event["indices"], event["feature"]] = original_values
            continue
        labels.update(event_labels)
        realized_strengths.extend(event_strengths)
    return injected, labels, realized_strengths
