"""Punkt- und Ereignismetriken des synthetischen Benchmarks."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from .benchmark_injections import ANOMALY_TYPES
from .detector import detector_output, event_ids


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


def _evaluate_injection_run(
    run, base, detector, base_alarm, include_point_metrics=True,
):
    data = base["data"]
    injected_alarm, _, _ = detector_output(
        run["score"], data["sensor"], data["time"],
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
            run["score"][observed],
        )
    return result


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
        base["score"], data["sensor"], data["time"],
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
