"""Evaluiert Forecasts und schreibt nachvollziehbare Anomalie-Kandidaten.

Der synthetische Vergleich aller Modelle liegt separat in ``src.benchmark``.
Optional können bestätigte Labels als CSV mit ``sensor,time,label`` übergeben
werden. Ohne solche Labels sind die Treffer Kandidaten, keine Ground Truth.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from .common import (
    ANOMALY_FEATURE, DATA_PATH, FEATURES, MODEL_DIR, anomaly_scores,
    conformal_bounds, inverse_values, load_data, make_windows, predict_quantiles,
    split_masks,
)
from ..detector import detector_output, event_ids


def _forecast_metrics(actual, forecast, baseline, lower, upper, mase_scale):
    result = {}
    for i, feature in enumerate(FEATURES):
        error = actual[:, i] - forecast[:, i]
        mae = np.mean(np.abs(error))
        baseline_mae = np.mean(np.abs(actual[:, i] - baseline[:, i]))
        result[feature] = {
            "mae": float(mae),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mase": float(mae / max(mase_scale[i], 1e-9)),
            "relative_mae_vs_t_minus_24_test": float(
                mae / max(baseline_mae, 1e-9)
            ),
            "seasonal_naive_test_mae": float(baseline_mae),
            "mase_training_scale": float(mase_scale[i]),
            "interval_coverage": float(np.mean(
                (actual[:, i] >= lower[:, i])
                & (actual[:, i] <= upper[:, i])
            )),
            "mean_interval_width": float(np.mean(upper[:, i] - lower[:, i])),
        }
    return result


def _classification_metrics(labels, alarms, scores):
    labels, alarms = np.asarray(labels, bool), np.asarray(alarms, bool)
    tn, fp, fn, tp = confusion_matrix(labels, alarms, labels=[False, True]).ravel()
    result = {
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "precision": float(precision_score(labels, alarms, zero_division=0)),
        "recall": float(recall_score(labels, alarms, zero_division=0)),
        "f1": float(f1_score(labels, alarms, zero_division=0)),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc"] = result["pr_auc"] = None
    return result


def _evaluate_labels(path, candidates):
    labels = pd.read_csv(path)
    if "timestamp" in labels and "time" not in labels:
        labels = labels.rename(columns={"timestamp": "time"})
    if not {"sensor", "time", "label"}.issubset(labels.columns):
        raise ValueError("Label-CSV benötigt sensor,time,label")
    filled = labels["label"].notna() & labels["label"].astype(str).str.strip().ne("")
    labels = labels.loc[filled].copy()
    if labels.empty:
        raise ValueError("Die Label-CSV enthält keine ausgefüllten Labels")
    labels["time"] = pd.to_datetime(labels["time"], utc=True)
    truth = labels["label"].astype(str).str.strip().str.lower().map({
        "1": True, "1.0": True, "true": True, "anomaly": True,
        "0": False, "0.0": False, "false": False, "normal": False,
    })
    if truth.isna().any():
        raise ValueError("label muss 0/1, true/false oder normal/anomaly sein")
    labels["label"] = truth.astype(bool)
    joined = candidates.merge(labels, on=["sensor", "time"], how="inner")
    if joined.empty:
        raise ValueError("Keine Labels passen zum Testzeitraum")
    return {
        "matched_rows": len(joined),
        "metrics": _classification_metrics(
            joined["label"], joined["anomaly"],
            joined["detector_score"].fillna(1.0).to_numpy(),
        ),
    }


def _event_table(candidates, event_column, alert_type):
    table = candidates.dropna(subset=[event_column]).groupby(event_column).agg(
        sensor=("sensor", "first"),
        start=("time", "min"),
        end=("time", "max"),
        duration_hours=("time", "size"),
        trigger_features=("trigger_features", lambda values: ",".join(
            dict.fromkeys(value for value in values if value)
        )),
        reason=("reason", lambda values: " | ".join(
            dict.fromkeys(value for value in values if value)
        )),
        max_anomaly_score=("anomaly_score", "max"),
        min_empirical_p_value=("p_value", "min"),
        max_context_imputed_fraction=("context_imputed_fraction", "max"),
    ).reset_index(drop=True)
    table.insert(0, "event_id", [
        f"{alert_type}-{i + 1}" for i in range(len(table))
    ])
    table.insert(1, "alert_type", alert_type)
    return table


def _save_example_plot(candidates, model_dir):
    plot = candidates[candidates["actual_visitors"].notna()]
    if plot.empty:
        return
    sensor = plot.groupby("sensor")["forecast_anomaly"].sum().idxmax()
    plot = plot[plot["sensor"] == sensor]
    plt.figure(figsize=(12, 4))
    plt.fill_between(
        plot["time"], plot["lower_visitors"], plot["upper_visitors"],
        alpha=0.2, label="Intervall",
    )
    plt.plot(plot["time"], plot["actual_visitors"], label="Ist")
    plt.plot(plot["time"], plot["forecast_visitors"], label="Prognose")
    hits = plot[plot["forecast_anomaly"]]
    plt.scatter(
        hits["time"], hits["actual_visitors"], color="red",
        label="Kandidat", zorder=3,
    )
    plt.title(sensor)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "anomalies_example.png", dpi=150)
    plt.close()


def evaluate(data_path=DATA_PATH, model_dir=MODEL_DIR, labels_path=None):
    model_dir = Path(model_dir)
    with open(model_dir / "model_bundle.json", encoding="utf-8") as file:
        bundle = json.load(file)
    if "detector" not in bundle:
        raise ValueError("Zuerst den gemeinsamen Lauf starten: python -m src.benchmark")
    df = load_data(data_path)
    data = make_windows(df, bundle["scaler"])
    masks = split_masks(data)
    test = {name: values[masks["test_all"]] for name, values in data.items()}
    observed = test["observed"]

    model = tf.keras.models.load_model(
        model_dir / bundle["model"], compile=False
    )
    quantiles = predict_quantiles(model, test)
    raw_quantiles = inverse_values(
        quantiles, test["sensor_id"], bundle["scaler"]
    )
    lower_scaled, upper_scaled = conformal_bounds(
        quantiles, bundle["calibrator"]
    )
    lower = inverse_values(lower_scaled, test["sensor_id"], bundle["scaler"])
    upper = inverse_values(upper_scaled, test["sensor_id"], bundle["scaler"])

    feature_score = np.full((len(observed), len(FEATURES)), np.nan)
    anomaly_score = np.full(len(observed), np.nan)
    p_value = np.full(len(observed), np.nan)
    scored = anomaly_scores(
        test["y_scaled"][observed], quantiles[observed], bundle["calibrator"]
    )
    feature_score[observed], anomaly_score[observed], p_value[observed] = scored
    forecast_alarm, persistence_count, detector_score = detector_output(
        anomaly_score, test["sensor"], test["time"], observed, bundle["detector"]
    )
    missing_alarm = ~observed

    candidates = pd.DataFrame({
        "sensor": test["sensor"],
        "time": pd.to_datetime(test["time"], utc=True),
        "context_imputed_fraction": test["imputed_fraction"],
        "anomaly_score": anomaly_score,
        "p_value": p_value,
        "persistence_count": persistence_count,
        "detector_score": detector_score,
        "forecast_anomaly": forecast_alarm,
        "missing_measurement": missing_alarm,
        "anomaly": forecast_alarm | missing_alarm,
    })
    for i, feature in enumerate(FEATURES):
        candidates[f"actual_{feature}"] = np.where(
            observed, test["y_raw"][:, i], np.nan
        )
        candidates[f"forecast_{feature}"] = raw_quantiles[:, i, 1]
        candidates[f"lower_{feature}"] = lower[:, i]
        candidates[f"upper_{feature}"] = upper[:, i]
        candidates[f"score_{feature}"] = feature_score[:, i]

    trigger = []
    reasons = []
    for row in range(len(candidates)):
        direct = [
            feature for i, feature in enumerate(FEATURES)
            if feature == ANOMALY_FEATURE and observed[row]
            and feature_score[row, i] > bundle["detector"]["point_threshold"]
        ]
        trigger.append(",".join(direct))
        if forecast_alarm[row] and not trigger[-1]:
            trigger[-1] = "persistence"
        if not observed[row]:
            reasons.append("Messung fehlt")
        elif direct:
            details = []
            for feature in direct:
                i = FEATURES.index(feature)
                actual = test["y_raw"][row, i]
                boundary = lower[row, i] if actual < lower[row, i] else upper[row, i]
                direction = "unter" if actual < lower[row, i] else "über"
                details.append(
                    f"{feature}: {actual:.3f} {direction} Intervallgrenze {boundary:.3f}"
                )
            reasons.append("; ".join(details))
        elif forecast_alarm[row]:
            reasons.append(
                f"Persistenz: {int(persistence_count[row])} von "
                f"{bundle['detector']['persistence_window']} Stunden auffällig"
            )
        else:
            reasons.append("")
    candidates["trigger_features"] = trigger
    candidates["reason"] = reasons
    candidates["alert_type"] = np.where(
        forecast_alarm, "forecast", np.where(missing_alarm, "missing", "")
    )
    candidates["forecast_event_id"] = event_ids(
        candidates.assign(anomaly=forecast_alarm)
    )
    candidates["missing_event_id"] = event_ids(
        candidates.assign(anomaly=missing_alarm)
    )

    forecast = _forecast_metrics(
        test["y_raw"][observed], raw_quantiles[observed, :, 1],
        test["seasonal"][observed], lower[observed], upper[observed],
        np.mean(np.abs(
            data["y_raw"][masks["train"]] - data["seasonal"][masks["train"]]
        ), axis=0),
    )
    joint_coverage = float(np.mean(
        ((test["y_raw"][observed] >= lower[observed])
         & (test["y_raw"][observed] <= upper[observed])).all(axis=1)
    ))
    label_result = _evaluate_labels(labels_path, candidates) if labels_path else None
    metrics = {
        "forecast": forecast,
        "all_feature_interval_coverage": joint_coverage,
        "anomaly_feature": ANOMALY_FEATURE,
        "detector": {
            "config": bundle["detector"],
            "calibration_selection": bundle["detector_selection"]["selected"],
        },
        "counts": {
            "measured_test_points": int(observed.sum()),
            "forecast_candidates": int(forecast_alarm.sum()),
            "missing_measurements": int(missing_alarm.sum()),
            "forecast_events": int(candidates["forecast_event_id"].nunique()),
            "missing_events": int(candidates["missing_event_id"].nunique()),
        },
        "real_labels": label_result,
        "limitation": (
            "Ohne bestätigte Ereignis- oder Sensorlabels sind statistische "
            "Abweichungen nur Anomalie-Kandidaten."
        ),
    }

    candidates.to_csv(model_dir / "anomalies.csv", index=False)
    candidates[["sensor", "time"]].assign(
        label="", label_type="", comment=""
    ).to_csv(model_dir / "labels_template.csv", index=False)
    pd.concat([
        _event_table(candidates, "forecast_event_id", "forecast"),
        _event_table(candidates, "missing_event_id", "missing"),
    ], ignore_index=True).to_csv(model_dir / "anomaly_events.csv", index=False)
    with open(
        model_dir / "evaluation_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=False)
    _save_example_plot(candidates, model_dir)

    print(
        f"Forecast-Kandidaten: {int(forecast_alarm.sum())} Punkte in "
        f"{int(candidates['forecast_event_id'].nunique())} Ereignissen"
    )
    print(f"Fehlende Messungen: {int(missing_alarm.sum())}")
    print(f"Intervallabdeckung beider Prognosegrößen: {joint_coverage:.1%}")
    print("Gemeinsamer Modellvergleich: python -m src.benchmark")
    print(f"Ereignisdetails: {model_dir / 'anomaly_events.csv'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.model_dir, args.labels)


if __name__ == "__main__":
    main()
