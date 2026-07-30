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

from ..benchmark_metrics import (
    _classification_metrics as _binary_metrics, _ranking_metrics,
)
from ..detector import detector_output, event_ids
from .common import (
    ANOMALY_FEATURE, DATA_PATH, FEATURES, MODEL_DIR, anomaly_scores,
    conformal_bounds, inverse_values, load_data, make_windows, predict_quantiles,
    split_masks,
)


def _forecast_metrics(actual, forecast, baseline, lower, upper, mase_scale):
    result = {}
    for i, feature in enumerate(FEATURES):
        error = actual[:, i] - forecast[:, i]
        mae = np.mean(np.abs(error))
        baseline_pairs = np.isfinite(baseline[:, i])
        if not baseline_pairs.any():
            raise ValueError(f"Keine gemessenen t-24-Paare für {feature}")
        paired_model_mae = np.mean(np.abs(error[baseline_pairs]))
        baseline_mae = np.mean(np.abs(
            actual[baseline_pairs, i] - baseline[baseline_pairs, i]
        ))
        result[feature] = {
            "mae": float(mae),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mase": float(mae / max(mase_scale[i], 1e-9)),
            "relative_mae_vs_t_minus_24_test": float(
                paired_model_mae / max(baseline_mae, 1e-9)
            ),
            "model_mae_on_t_minus_24_pairs": float(paired_model_mae),
            "seasonal_naive_test_mae": float(baseline_mae),
            "seasonal_naive_test_pairs": int(baseline_pairs.sum()),
            "mase_training_scale": float(mase_scale[i]),
            "interval_coverage": float(np.mean(
                (actual[:, i] >= lower[:, i])
                & (actual[:, i] <= upper[:, i])
            )),
            "mean_interval_width": float(np.mean(upper[:, i] - lower[:, i])),
        }
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
    if labels.duplicated(["sensor", "time"]).any():
        raise ValueError("Label-CSV enthält Sensor/Zeit mehrfach")

    measured = candidates.loc[~candidates["data_quality_issue"]].copy()
    joined = measured.merge(
        labels, on=["sensor", "time"], how="inner", validate="one_to_one"
    )
    if joined.empty:
        raise ValueError("Keine Labels passen zum Testzeitraum")
    complete = len(joined) == len(measured)
    metrics = None
    if complete:
        metrics = _binary_metrics(joined["label"], joined["forecast_anomaly"])
        ranking = _ranking_metrics(joined["label"], joined["anomaly_score"])
        metrics.update({key: ranking[key] for key in (
            "roc_auc", "average_precision"
        )})
    return {
        "matched_rows": len(joined),
        "coverage": {
            "labeled_measured_points": int(len(joined)),
            "measured_test_points": int(len(measured)),
            "fraction": float(len(joined) / len(measured)) if len(measured) else 0.0,
            "complete": bool(complete),
        },
        "warning": None if complete else (
            "Labels decken nicht die vollständige gemessene Testpopulation ab; "
            "Precision, Recall, F1 und AUC werden deshalb nicht berechnet."
        ),
        "metrics": metrics,
    }


def _alert_confidence(context_fraction, actual, lower, upper):
    excess = max(lower - actual, actual - upper, 0.0)
    notes = []
    if context_fraction > 0.5:
        notes.append("mehr als 50 % des Kontexts imputiert")
    if excess < 1.0:
        notes.append(
            f"Überschreitung der Besucher-Intervallgrenze nur {excess:.3f} "
            "(< 1 Besucher)"
        )
    if notes:
        return "low", "Niedrige Konfidenz: " + "; ".join(notes)
    return "normal", ""


def _unmodeled_missing_candidates(df, modeled_sensors, test_start):
    missing = df[
        ~df["name"].isin(modeled_sensors)
        & (df["timestamp"] >= test_start)
        & df["is_imputed"]
    ]
    return pd.DataFrame({
        "sensor": missing["name"].to_numpy(),
        "time": missing["timestamp"].to_numpy(),
        "forecast_anomaly": False, "missing_measurement": True,
        "data_quality_issue": True, "anomaly": False, "alert": True,
        "trigger_features": "",
        "reason": "Messung fehlt; kein Forecast wegen zu kurzer Trainingshistorie",
        "confidence": "", "alert_type": "missing",
    })


def _event_review_template(events, path):
    fields = ["review_status", "label", "label_type", "comment", "source_url"]
    if Path(path).exists():
        old = pd.read_csv(path)
        keys = ["sensor", "start", "end"]
        if not set(keys + fields).issubset(old):
            raise ValueError("Bestehende Event-Review-Datei hat falsche Spalten")
        old["start"] = pd.to_datetime(old["start"], utc=True)
        old["end"] = pd.to_datetime(old["end"], utc=True)
        events = events.merge(
            old[keys + fields], on=keys, how="left", validate="one_to_one"
        )
    for field in fields:
        default = "offen" if field == "review_status" else ""
        events[field] = events[field].fillna(default) if field in events else default
    return events


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
        confidence=("confidence", "min"),
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
    all_df = load_data(data_path, min_train_points=None)
    modeled_sensors = bundle["scaler"]["sensors"]
    df = all_df[all_df["name"].isin(modeled_sensors)].copy()
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
        "data_quality_issue": missing_alarm,
        "anomaly": forecast_alarm,
        "alert": forecast_alarm | missing_alarm,
    })
    for i, feature in enumerate(FEATURES):
        candidates[f"actual_{feature}"] = np.where(
            observed, test["y_raw"][:, i], np.nan
        )
        candidates[f"forecast_{feature}"] = raw_quantiles[:, i, 1]
        candidates[f"lower_{feature}"] = lower[:, i]
        candidates[f"upper_{feature}"] = upper[:, i]
        candidates[f"score_{feature}"] = feature_score[:, i]

    visitor_index = FEATURES.index(ANOMALY_FEATURE)
    direct = observed & (
        feature_score[:, visitor_index] > bundle["detector"]["point_threshold"]
    )
    candidates["trigger_features"] = np.where(
        direct, ANOMALY_FEATURE, np.where(forecast_alarm, "persistence", "")
    )
    reasons = np.where(observed, "", "Messung fehlt").astype(object)
    confidence = np.full(len(observed), "", dtype=object)
    for row in np.flatnonzero(forecast_alarm):
        if direct[row]:
            actual = test["y_raw"][row, visitor_index]
            below = actual < lower[row, visitor_index]
            boundary = lower[row, visitor_index] if below else upper[row, visitor_index]
            reason = (
                f"{ANOMALY_FEATURE}: {actual:.3f} "
                f"{'unter' if below else 'über'} Intervallgrenze {boundary:.3f}"
            )
        else:
            reason = (
                f"Persistenz: {int(persistence_count[row])} von "
                f"{bundle['detector']['persistence_window']} Stunden auffällig"
            )
        confidence[row], note = _alert_confidence(
            test["imputed_fraction"][row], test["y_raw"][row, visitor_index],
            lower[row, visitor_index], upper[row, visitor_index],
        )
        reasons[row] = "; ".join(filter(None, [reason, note]))
    candidates["reason"] = reasons
    candidates["confidence"] = confidence
    candidates["alert_type"] = np.where(
        forecast_alarm, "forecast", np.where(missing_alarm, "missing", "")
    )
    unmodeled_missing = _unmodeled_missing_candidates(
        all_df, modeled_sensors, pd.Timestamp(bundle["splits"]["calibration_end"]),
    )
    candidates = pd.concat(
        [candidates, unmodeled_missing], ignore_index=True
    ).sort_values(["sensor", "time"]).reset_index(drop=True)
    candidates["forecast_event_id"] = event_ids(
        candidates.assign(anomaly=candidates["forecast_anomaly"])
    )
    candidates["missing_event_id"] = event_ids(
        candidates.assign(anomaly=candidates["missing_measurement"])
    )

    seasonal = test["seasonal"][observed].copy()
    seasonal[~test["seasonal_observed"][observed]] = np.nan
    train_pairs = masks["train"] & data["seasonal_observed"]
    forecast = _forecast_metrics(
        test["y_raw"][observed], raw_quantiles[observed, :, 1],
        seasonal, lower[observed], upper[observed],
        np.mean(np.abs(
            data["y_raw"][train_pairs] - data["seasonal"][train_pairs]
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
            "modeled_sensors": len(modeled_sensors),
            "data_quality_sensors": int(all_df["name"].nunique()),
            "measured_test_points": int(observed.sum()),
            "forecast_candidates": int(forecast_alarm.sum()),
            "missing_measurements": int(candidates["missing_measurement"].sum()),
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
    candidates.loc[
        ~candidates["data_quality_issue"],
        ["sensor", "time", "forecast_anomaly", "reason", "confidence"],
    ].assign(label="", label_type="", comment="", source_url="").to_csv(
        model_dir / "labels_template.csv", index=False
    )
    forecast_events = _event_table(
        candidates, "forecast_event_id", "forecast"
    )
    missing_events = _event_table(candidates, "missing_event_id", "missing")
    pd.concat([forecast_events, missing_events], ignore_index=True).to_csv(
        model_dir / "anomaly_events.csv", index=False
    )
    _event_review_template(
        forecast_events, model_dir / "event_review_template.csv"
    ).to_csv(model_dir / "event_review_template.csv", index=False)
    with open(
        model_dir / "evaluation_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=False)
    _save_example_plot(candidates, model_dir)

    print(
        f"Forecast-Kandidaten: {int(forecast_alarm.sum())} Punkte in "
        f"{int(candidates['forecast_event_id'].nunique())} Ereignissen"
    )
    print(f"Fehlende Messungen: {int(candidates['missing_measurement'].sum())}")
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
