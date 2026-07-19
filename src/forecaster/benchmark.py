"""Wählt den Detektor und prüft ihn mit synthetischen Anomalien.

Normaler Aufruf des separaten Benchmarks aus dem Projektordner:
``python -m src.forecaster.benchmark``
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from .common import (
    CALIBRATION_END, DATA_PATH, FEATURES, MODEL_DIR, TUNE_END, anomaly_scores,
    inverse_values, load_data, make_windows, predict_quantiles, scale_values,
)
from .detector import detector_output, event_ids


HIGH_ALPHAS = [0.005, 0.01]
LOW_ALPHAS = [0.05, 0.10]
PERSISTENCE_RULES = [(3, 2), (4, 3), (6, 3), (6, 4)]
MAX_CALIBRATION_ALARM_RATE = 0.01
CALIBRATION_SEEDS = [142, 243, 344]
TEST_SEEDS = [445, 546, 647]
ANOMALY_TYPES = ["spike", "dip", "level_shift", "drift", "stuck_at"]


def _classification_metrics(labels, alarms, scores):
    labels, alarms = np.asarray(labels, bool), np.asarray(alarms, bool)
    tn, fp, fn, tp = confusion_matrix(labels, alarms, labels=[False, True]).ravel()
    union = tp + fp + fn
    result = {
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "precision": float(precision_score(labels, alarms, zero_division=0)),
        "recall": float(recall_score(labels, alarms, zero_division=0)),
        "f1": float(f1_score(labels, alarms, zero_division=0)),
        "temporal_iou": float(tp / union) if union else 1.0,
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc"] = result["pr_auc"] = None
    return result


def _empirical_threshold(scores, alpha):
    values = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    if not len(values):
        raise ValueError("Für die Schwelle liegen keine endlichen Scores vor")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1 - alpha))))
    return float(values[rank - 1])


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
            "feature": FEATURES[number % len(FEATURES)],
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
            event_labels.add((row["name"], row["timestamp"].to_datetime64()))
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
            event_strengths.append(abs(float(actual_scaled) - original))

        if anomaly_type == "stuck_at" and np.mean(event_strengths) < 0.5:
            injected.loc[event["indices"], event["feature"]] = original_values
            continue
        labels.update(event_labels)
        realized_strengths.extend(event_strengths)
    return injected, labels, realized_strengths


def _event_metrics(frame):
    """Berechnet Ereignismetriken ohne nachtraegliches Point-Adjustment."""
    true_ids = event_ids(frame.rename(columns={"label": "anomaly"}))
    pred_ids = event_ids(frame.rename(columns={"prediction": "anomaly"}))
    true_events = [
        group for _, group in frame.assign(event=true_ids).dropna(
            subset=["event"]
        ).groupby("event")
    ]
    pred_events = [
        group for _, group in frame.assign(event=pred_ids).dropna(
            subset=["event"]
        ).groupby("event")
    ]

    matches, overlaps, delays = 0, [], []
    for event in true_events:
        sensor = event["sensor"].iloc[0]
        truth = set(event["time"])
        best, first = 0.0, None
        for predicted in pred_events:
            if predicted["sensor"].iloc[0] != sensor:
                continue
            found = set(predicted["time"])
            common = truth & found
            if common:
                best = max(best, len(common) / len(truth | found))
                first = min(common) if first is None else min(first, min(common))
        if first is not None:
            matches += 1
            delays.append((first - min(truth)).total_seconds() / 3600)
        overlaps.append(best)

    matched_predictions = sum(
        any(
            predicted["sensor"].iloc[0] == truth["sensor"].iloc[0]
            and bool(set(predicted["time"]) & set(truth["time"]))
            for truth in true_events
        )
        for predicted in pred_events
    )
    precision = matched_predictions / len(pred_events) if pred_events else 0.0
    recall = matches / len(true_events) if true_events else 0.0
    return {
        "true_events": len(true_events),
        "predicted_events": len(pred_events),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "mean_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_delay_hours": float(np.mean(delays)) if delays else None,
    }


def _period_scores(data, model, bundle, start, end=None):
    time = pd.to_datetime(data["time"], utc=True)
    mask = time >= start
    if end is not None:
        mask &= time < end
    period = {name: values[np.asarray(mask)] for name, values in data.items()}
    quantiles = predict_quantiles(model, period)
    joint_score = np.full(len(period["time"]), np.nan)
    observed = period["observed"]
    _, joint_score[observed], _ = anomaly_scores(
        period["y_scaled"][observed], quantiles[observed], bundle["calibrator"]
    )
    return period, joint_score


def _prepare_injection_runs(
    df, base_data, model, bundle, start, end, seeds, include_mild,
):
    """Berechnet die Scores einmal je Injektionslauf."""
    base, base_joint = _period_scores(base_data, model, bundle, start, end)
    base_keys = [
        (sensor, np.datetime64(time, "us"))
        for sensor, time in zip(base["sensor"], base["time"])
    ]
    runs = []
    for seed in seeds:
        for type_id, anomaly_type in enumerate(ANOMALY_TYPES):
            events = _injection_events(
                df, anomaly_type, start, end, seed + 1000 * type_id
            )
            strengths = [None] if anomaly_type == "stuck_at" else (
                [0.5, 1.0, 2.0] if include_mild else [1.0, 2.0]
            )
            duration = 1 if anomaly_type in {"spike", "dip"} else (
                6 if anomaly_type == "stuck_at" else 4
            )
            for strength in strengths:
                injected_df, label_keys, realized = _inject(
                    df, events, anomaly_type, strength or 0, bundle["scaler"]
                )
                if not label_keys:
                    continue
                injected_data = make_windows(injected_df, bundle["scaler"])
                injected, joint = _period_scores(
                    injected_data, model, bundle, start, end
                )
                injected_keys = [
                    (sensor, np.datetime64(time, "us"))
                    for sensor, time in zip(injected["sensor"], injected["time"])
                ]
                if base_keys != injected_keys:
                    raise ValueError("Injection hat die Zeitachse verändert")
                runs.append({
                    "anomaly_type": anomaly_type,
                    "strength": "6_hours" if strength is None else str(strength),
                    "event_count": len(label_keys) // duration,
                    "realized_strength": float(np.mean(realized)),
                    "joint_score": joint,
                    "labels": np.array([key in label_keys for key in base_keys]),
                })
    return {"data": base, "joint_score": base_joint}, runs


def _evaluate_injection_run(
    run, base, detector, base_alarm, include_point_metrics=True,
):
    data = base["data"]
    injected_alarm, _, _ = detector_output(
        run["joint_score"], data["sensor"], data["time"],
        data["observed"], detector,
    )
    incremental = injected_alarm & ~base_alarm
    observed = data["observed"]
    frame = pd.DataFrame({
        "sensor": data["sensor"][observed],
        "time": pd.to_datetime(data["time"][observed], utc=True),
        "label": run["labels"][observed],
        "prediction": incremental[observed],
    })
    result = {"event": _event_metrics(frame)}
    if include_point_metrics:
        result["point"] = _classification_metrics(
            run["labels"][observed], incremental[observed],
            run["joint_score"][observed],
        )
    return result


def _detector_candidates(base):
    data, scores = base["data"], base["joint_score"]
    observed_scores = scores[data["observed"]]
    for high_alpha in HIGH_ALPHAS:
        high = _empirical_threshold(observed_scores, high_alpha)
        yield {
            "high_alpha": high_alpha,
            "point_threshold": high,
            "low_alpha": None,
            "low_threshold": None,
            "persistence_window": None,
            "required_exceedances": None,
        }
        for low_alpha in LOW_ALPHAS:
            low = _empirical_threshold(observed_scores, low_alpha)
            for window, required in PERSISTENCE_RULES:
                yield {
                    "high_alpha": high_alpha,
                    "point_threshold": high,
                    "low_alpha": low_alpha,
                    "low_threshold": low,
                    "persistence_window": window,
                    "required_exceedances": required,
                }


def select_detector(df, base_data, model, bundle):
    """Wählt die Alarmregel nur mit Kalibrierungsinjektionen."""
    base, runs = _prepare_injection_runs(
        df, base_data, model, bundle, TUNE_END, CALIBRATION_END,
        CALIBRATION_SEEDS, include_mild=False,
    )
    data = base["data"]
    observed = data["observed"]
    reports = []
    for detector in _detector_candidates(base):
        base_alarm, _, _ = detector_output(
            base["joint_score"], data["sensor"], data["time"], observed, detector
        )
        rate = float(base_alarm[observed].mean())
        if rate > MAX_CALIBRATION_ALARM_RATE:
            continue
        by_type = {name: [] for name in ANOMALY_TYPES}
        for run in runs:
            event = _evaluate_injection_run(
                run, base, detector, base_alarm, include_point_metrics=False
            )["event"]
            by_type[run["anomaly_type"]].append(event)
        type_f1 = {
            name: float(np.mean([value["f1"] for value in values]))
            for name, values in by_type.items() if values
        }
        type_recall = {
            name: float(np.mean([value["recall"] for value in values]))
            for name, values in by_type.items() if values
        }
        reports.append({
            "detector": detector,
            "calibration_alarm_count": int(base_alarm[observed].sum()),
            "calibration_alarm_rate": rate,
            "macro_event_f1": float(np.mean(list(type_f1.values()))),
            "macro_incremental_event_recall": float(
                np.mean(list(type_recall.values()))
            ),
            "event_f1_by_type": type_f1,
        })
    if not reports:
        raise ValueError("Keine Detektorkonfiguration erfüllt das Alarmratenbudget")

    def order(report):
        detector = report["detector"]
        return (
            report["macro_event_f1"],
            report["macro_incremental_event_recall"],
            -report["calibration_alarm_rate"],
            -(detector["persistence_window"] or 0),
        )

    selected = max(reports, key=order)
    detector = {
        **selected["detector"],
        "selection_period": [str(TUNE_END), str(CALIBRATION_END)],
        "max_calibration_alarm_rate": MAX_CALIBRATION_ALARM_RATE,
    }
    selection = {
        "seeds": CALIBRATION_SEEDS,
        "selection_strengths": [1.0, 2.0],
        "objective": "macro incremental event F1, equal weight per anomaly type",
        "selected": selected,
        "feasible_candidates": reports,
    }
    return detector, selection


def _mean_result(results, section, keys):
    return {
        key: (None if all(result[section].get(key) is None for result in results)
              else float(np.mean([
                  result[section][key] for result in results
                  if result[section].get(key) is not None
              ])))
        for key in keys
    }


def _aggregate_injection_runs(runs, base, detector):
    data = base["data"]
    base_alarm, _, _ = detector_output(
        base["joint_score"], data["sensor"], data["time"],
        data["observed"], detector,
    )
    result = {}
    for anomaly_type in ANOMALY_TYPES:
        type_runs = [run for run in runs if run["anomaly_type"] == anomaly_type]
        type_result = {"one_event_per_sensor": True, "strengths": {}}
        for strength in sorted({run["strength"] for run in type_runs}):
            group = [run for run in type_runs if run["strength"] == strength]
            evaluated = [
                _evaluate_injection_run(run, base, detector, base_alarm)
                for run in group
            ]
            type_result["strengths"][strength] = {
                "runs": len(group),
                "mean_events_per_run": float(np.mean([
                    run["event_count"] for run in group
                ])),
                "mean_realized_log_iqr_strength": float(np.mean([
                    run["realized_strength"] for run in group
                ])),
                "point": _mean_result(
                    evaluated, "point",
                    ["precision", "recall", "f1", "temporal_iou",
                     "roc_auc", "pr_auc"],
                ),
                "event": _mean_result(
                    evaluated, "event",
                    ["precision", "recall", "f1", "mean_iou",
                     "mean_delay_hours"],
                ),
            }
        result[anomaly_type] = type_result
    return result


def _reference_detector(bundle):
    return {
        "high_alpha": bundle["calibrator"]["alpha"],
        "point_threshold": bundle["calibrator"]["threshold"],
        "low_alpha": None,
        "low_threshold": None,
        "persistence_window": None,
        "required_exceedances": None,
    }


def injection_benchmark(df, base_data, model, bundle):
    """Prüft die eingefrorene Alarmregel mit unabhängigen Testinjektionen."""
    base, runs = _prepare_injection_runs(
        df, base_data, model, bundle, CALIBRATION_END, None,
        TEST_SEEDS, include_mild=True,
    )
    return (
        _aggregate_injection_runs(runs, base, bundle["detector"]),
        _aggregate_injection_runs(runs, base, _reference_detector(bundle)),
    )


def _benchmark_rows(results, reference):
    rows = []
    for anomaly_type, type_result in results.items():
        for strength, result in type_result["strengths"].items():
            baseline = reference[anomaly_type]["strengths"][strength]
            rows.append({
                "anomaly_type": anomaly_type,
                "strength": strength,
                "runs": result["runs"],
                "mean_events_per_run": result["mean_events_per_run"],
                "mean_realized_log_iqr_strength": result[
                    "mean_realized_log_iqr_strength"
                ],
                **{f"point_{key}": result["point"][key]
                   for key in ["precision", "recall", "f1", "pr_auc"]},
                **{f"event_{key}": result["event"][key]
                   for key in ["precision", "recall", "f1", "mean_iou",
                               "mean_delay_hours"]},
                **{f"reference_point_{key}": baseline["point"][key]
                   for key in ["precision", "recall", "f1", "pr_auc"]},
                **{f"reference_event_{key}": baseline["event"][key]
                   for key in ["precision", "recall", "f1", "mean_iou",
                               "mean_delay_hours"]},
            })
    return rows


def run_benchmark(data_path=DATA_PATH, model_dir=MODEL_DIR):
    """Führt nur den synthetischen Benchmark aus und speichert dessen Resultate."""
    model_dir = Path(model_dir)
    with open(model_dir / "model_bundle.json", encoding="utf-8") as file:
        bundle = json.load(file)
    df = load_data(data_path)
    data = make_windows(df, bundle["scaler"])
    model = tf.keras.models.load_model(
        model_dir / bundle["model"], compile=False
    )
    results, reference = injection_benchmark(df, data, model, bundle)
    report = {
        "design": (
            "Injektion nur in ursprünglich gemessene Testwerte vor Windowing "
            "und Prediction; drei von der Kalibrierung getrennte Seeds; bewertet "
            "werden nur zusätzliche Alarme gegenüber dem unveränderten Lauf."
        ),
        "test_seeds": TEST_SEEDS,
        "warning": (
            "Die Ground Truth ist synthetisch. Der Recall gilt nur für die "
            "injizierten Typen und Stärken, nicht für unbekannte reale Anomalien."
        ),
        "results": results,
        "reference_one_percent_detector": reference,
    }
    with open(
        model_dir / "injection_benchmark.json", "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    pd.DataFrame(_benchmark_rows(results, reference)).to_csv(
        model_dir / "injection_benchmark.csv", index=False
    )
    print(f"Benchmark gespeichert: {model_dir / 'injection_benchmark.csv'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    run_benchmark(args.data, args.model_dir)


if __name__ == "__main__":
    main()
