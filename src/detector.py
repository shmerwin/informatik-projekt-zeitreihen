"""Gemeinsame Logik für Alarme und zeitliche Ereignisse."""

import numpy as np
import pandas as pd


def causal_exceedance_count(scores, sensors, times, observed, threshold, window):
    """Zählt Überschreitungen in den letzten lückenlos gemessenen Stunden."""
    scores = np.asarray(scores, float)
    sensors = np.asarray(sensors)
    times = np.asarray(times, dtype="datetime64[us]")
    observed = np.asarray(observed, bool)
    result = np.full(len(scores), np.nan)
    one_hour = np.timedelta64(1, "h")

    for sensor in pd.unique(sensors):
        positions = np.flatnonzero(sensors == sensor)
        history = []
        previous = None
        for position in positions:
            current = times[position]
            if (not observed[position] or previous is None
                    or current - previous != one_hour):
                history = []
            if observed[position] and np.isfinite(scores[position]):
                history.append(scores[position] > threshold)
                history = history[-window:]
                if len(history) == window:
                    result[position] = float(np.sum(history))
                previous = current
            else:
                previous = None
    return result


def detector_output(joint_score, sensors, times, observed, detector):
    """Wendet Punktschwelle und optionale kausale Persistenzregel an."""
    if detector.get("persistence_window"):
        count = causal_exceedance_count(
            joint_score, sensors, times, observed, detector["low_threshold"],
            detector["persistence_window"],
        )
        persistence = count - (detector["required_exceedances"] - 0.5)
    else:
        count = np.full(len(joint_score), np.nan)
        persistence = np.full(len(joint_score), -np.inf)

    evidence = np.fmax(
        np.asarray(joint_score) - detector["point_threshold"], persistence
    )
    alarm = np.asarray(observed, bool) & (evidence > 0)
    return alarm, count, evidence


def event_ids(frame):
    """Gruppiert aufeinanderfolgende Alarmstunden je Sensor zu Ereignissen."""
    ids = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    number = 0
    for _, group in frame.groupby("sensor", sort=False):
        previous = None
        for index, time in zip(
            group.index[group["anomaly"]],
            group.loc[group["anomaly"], "time"],
        ):
            if previous is None or time - previous != pd.Timedelta(hours=1):
                number += 1
            ids.loc[index] = number
            previous = time
    return ids
