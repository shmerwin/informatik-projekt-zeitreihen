"""Punkt- und Ereignismetriken des synthetischen Benchmarks."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from .benchmark_injections import ANOMALY_TYPES
from .detector import detector_output, event_ids


MAX_CURVE_POINTS = 101


def _classification_metrics(labels, alarms):
    """Bewertet nur die binaeren, zusaetzlichen Alarme."""
    labels, alarms = np.asarray(labels, bool), np.asarray(alarms, bool)
    tn, fp, fn, tp = confusion_matrix(labels, alarms, labels=[False, True]).ravel()
    union = tp + fp + fn
    return {
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "precision": float(precision_score(labels, alarms, zero_division=0)),
        "recall": float(recall_score(labels, alarms, zero_division=0)),
        "f1": float(f1_score(labels, alarms, zero_division=0)),
        "temporal_iou": float(tp / union) if union else 1.0,
    }


def _ranking_metrics(labels, scores, include_curves=False):
    """Bewertet den rohen Score unabhaengig von der gewaehlten Alarmregel."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    finite = np.isfinite(scores)
    result = {
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "missing_score_count": int((~finite).sum()),
        "roc_auc": None,
        "average_precision": None,
    }
    if finite.any():
        spread = max(float(np.ptp(scores[finite])), 1.0)
        scores = np.where(finite, scores, scores[finite].min() - spread)
    else:
        scores = np.zeros(len(scores))
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(
            average_precision_score(labels, scores)
        )
        if include_curves:
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)

            def compact(x, y):
                count = min(MAX_CURVE_POINTS, len(x))
                indices = np.unique(np.linspace(
                    0, len(x) - 1, count, dtype=int
                ))
                return np.asarray(x)[indices].tolist(), np.asarray(y)[indices].tolist()

            fpr, tpr = compact(false_positive_rate, true_positive_rate)
            pr_precision, pr_recall = compact(precision, recall)
            result["roc_curve"] = {
                "false_positive_rate": fpr,
                "true_positive_rate": tpr,
            }
            result["pr_curve"] = {
                "recall": pr_recall,
                "precision": pr_precision,
            }
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

    candidates = []
    for true_id, event in enumerate(true_events):
        truth = set(event["time"])
        for pred_id, predicted in enumerate(pred_events):
            if predicted["sensor"].iloc[0] != event["sensor"].iloc[0]:
                continue
            found = set(predicted["time"])
            common = truth & found
            if common:
                candidates.append((
                    len(common) / len(truth | found), true_id, pred_id,
                    min(common), min(truth),
                ))

    matched_true, matched_pred, matches = set(), set(), []
    for candidate in sorted(candidates, reverse=True):
        _, true_id, pred_id, _, _ = candidate
        if true_id not in matched_true and pred_id not in matched_pred:
            matched_true.add(true_id)
            matched_pred.add(pred_id)
            matches.append(candidate)

    overlaps = np.zeros(len(true_events))
    delays = []
    for overlap, true_id, _, first, start in matches:
        overlaps[true_id] = overlap
        delays.append((first - start).total_seconds() / 3600)

    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / len(pred_events) if pred_events else 0.0
    recall = tp / len(true_events) if true_events else 0.0
    return {
        "true_events": len(true_events),
        "predicted_events": len(pred_events),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "mean_iou": float(np.mean(overlaps)) if len(overlaps) else 0.0,
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
    result = {"incremental_event": _event_metrics(frame)}
    if include_point_metrics:
        result["labels"] = frame["label"].to_numpy()
        result["alarms"] = frame["prediction"].to_numpy()
    return result


def _aggregate_event_metrics(results):
    events = [result["incremental_event"] for result in results]
    tp, fp, fn = (sum(result[key] for result in events)
                  for key in ["tp", "fp", "fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    true_events = tp + fn
    return {
        "true_events": true_events,
        "predicted_events": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "mean_iou": (
            sum(result["mean_iou"] * result["true_events"] for result in events)
            / true_events if true_events else 0.0
        ),
        "mean_delay_hours": (
            sum(result["mean_delay_hours"] * result["tp"] for result in events
                if result["mean_delay_hours"] is not None) / tp
            if tp else None
        ),
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
            labels = np.concatenate([run["labels"] for run in evaluated])
            alarms = np.concatenate([run["alarms"] for run in evaluated])
            scores = np.concatenate([
                run["score"][data["observed"]] for run in group
            ])
            type_result["strengths"][strength] = {
                "runs": len(group),
                "mean_events_per_run": float(np.mean([
                    run["event_count"] for run in group
                ])),
                "mean_realized_log_iqr_strength": float(np.mean([
                    run["realized_strength"] for run in group
                ])),
                "incremental_point": _classification_metrics(labels, alarms),
                "incremental_event": _aggregate_event_metrics(evaluated),
                "raw_score_ranking": _ranking_metrics(
                    labels, scores, include_curves=True
                ),
            }
        result[anomaly_type] = type_result
    return result


def _benchmark_rows(results, reference):
    rows = []
    for anomaly_type in ANOMALY_TYPES:
        type_result = results[anomaly_type]
        for strength, result in type_result["strengths"].items():
            baseline = reference[anomaly_type]["strengths"][strength]
            ranking = result["raw_score_ranking"]
            row = {
                "anomaly_type": anomaly_type,
                "strength": strength,
                "runs": result["runs"],
                "mean_events_per_run": result["mean_events_per_run"],
                "mean_realized_log_iqr_strength": result[
                    "mean_realized_log_iqr_strength"
                ],
                **{f"raw_score_{key}": ranking[key]
                   for key in ["roc_auc", "average_precision"]},
            }
            for prefix, source in [("", result), ("reference_", baseline)]:
                point, event = source["incremental_point"], source["incremental_event"]
                row.update({
                    **{f"{prefix}incremental_point_{key}": point[key]
                       for key in ["precision", "recall", "f1", "temporal_iou"]},
                    **{f"{prefix}incremental_point_{key}":
                       point["confusion_matrix"][key]
                       for key in ["tn", "fp", "fn", "tp"]},
                    **{f"{prefix}incremental_event_{key}": event[key]
                       for key in ["precision", "recall", "f1", "mean_iou",
                                   "mean_delay_hours", "tp", "fp", "fn"]},
                })
            rows.append(row)
    return rows
