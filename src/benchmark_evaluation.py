"""Scoring, Detektorauswahl und Testläufe des synthetischen Benchmarks."""

import numpy as np
import pandas as pd

from .benchmark_injections import ANOMALY_TYPES, _inject, _injection_events
from .benchmark_metrics import _aggregate_injection_runs, _evaluate_injection_run
from .detector import detector_output
from .forecaster.common import (
    CALIBRATION_END, TUNE_END, WINDOW_SIZE, anomaly_scores, make_windows,
    predict_quantiles,
)


HIGH_ALPHAS = [0.005, 0.01]
LOW_ALPHAS = [0.05, 0.10]
PERSISTENCE_RULES = [(3, 2), (4, 3), (6, 3), (6, 4)]
MAX_CALIBRATION_ALARM_RATE = 0.01
CALIBRATION_SEEDS = [142, 243, 344]
TEST_SEEDS = [445, 546, 647]
NOISE_STRENGTHS = [0.05, 0.15]
NOISE_SEED = 748


def _empirical_threshold(scores, alpha):
    values = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    if not len(values):
        raise ValueError("Für die Schwelle liegen keine endlichen Scores vor")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1 - alpha))))
    return float(values[rank - 1])


def _forecast_scores(data, model, bundle):
    """Berechnet den vorhandenen Forecast-Score für ein Datenstück."""
    quantiles = predict_quantiles(model, data)
    result = np.full(len(data["time"]), np.nan)
    observed = data["observed"]
    _, result[observed], _ = anomaly_scores(
        data["y_scaled"][observed], quantiles[observed], bundle["calibrator"]
    )
    return result


def _period_scores(data, model, bundle, start, end=None, score_function=None):
    time = pd.to_datetime(data["time"], utc=True)
    mask = time >= start
    if end is not None:
        mask &= time < end
    period = {name: values[np.asarray(mask)] for name, values in data.items()}
    score_function = score_function or _forecast_scores
    return period, score_function(period, model, bundle)


def _prepare_injection_runs(
    df, base_data, model, bundle, start, end, seeds, include_mild,
    score_function=None,
):
    """Berechnet die Scores einmal je Injektionslauf."""
    base, base_score = _period_scores(
        base_data, model, bundle, start, end, score_function
    )
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    relevant = timestamps >= start - pd.Timedelta(hours=WINDOW_SIZE)
    if end is not None:
        relevant &= timestamps < end
    period_df = df.loc[relevant].copy()
    base_keys = [
        (sensor, np.datetime64(time, "us"))
        for sensor, time in zip(base["sensor"], base["time"])
    ]
    runs = []
    for seed in seeds:
        for type_id, anomaly_type in enumerate(ANOMALY_TYPES):
            events = _injection_events(
                period_df, anomaly_type, start, end, seed + 1000 * type_id
            )
            strengths = [None] if anomaly_type == "stuck_at" else (
                [0.5, 1.0, 2.0] if include_mild else [1.0, 2.0]
            )
            for strength in strengths:
                injected_df, label_keys, realized = _inject(
                    period_df, events, anomaly_type, strength or 0,
                    bundle["scaler"],
                )
                if not label_keys:
                    continue
                injected_data = make_windows(injected_df, bundle["scaler"])
                injected, score = _period_scores(
                    injected_data, model, bundle, start, end, score_function
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
                    "event_count": len({sensor for sensor, _ in label_keys}),
                    "realized_strength": float(np.mean(realized)),
                    "score": score,
                    "labels": np.array([key in label_keys for key in base_keys]),
                })
    return {"data": base, "score": base_score}, runs


def _add_count_noise(df, strength, seed):
    """Addiert mittelwerttreues, nichtnegatives Zaehlwertrauschen."""
    noisy = df.copy()
    time = pd.to_datetime(noisy["timestamp"], utc=True)
    mask = (time >= CALIBRATION_END) & ~noisy["is_imputed"]
    counts = np.rint(noisy.loc[mask, "visitors"]).astype("int64").clip(lower=0)
    rng = np.random.default_rng(seed)
    values = (
        rng.binomial(counts, 1 - strength)
        + rng.poisson(counts * strength)
    )
    noisy.loc[mask, "visitors"] = values
    return noisy


def _noise_robustness(df, base, model, bundle, score_function=None):
    data, base_score = base["data"], base["score"]
    base_keys = list(zip(base["data"]["sensor"], base["data"]["time"]))
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    period_df = df.loc[
        timestamps >= CALIBRATION_END - pd.Timedelta(hours=WINDOW_SIZE)
    ].copy()
    observed = data["observed"]
    base_alarm, _, _ = detector_output(
        base_score, data["sensor"], data["time"], observed, bundle["detector"]
    )
    denominator = int(observed.sum())
    if not denominator:
        raise ValueError("Der Testblock enthaelt keine gemessenen Zielwerte")
    result = {
        "observed_points": denominator,
        "base_alarm_count": int(base_alarm[observed].sum()),
        "base_alarm_rate": float(base_alarm[observed].mean()),
        "strengths": {},
    }
    for number, strength in enumerate(NOISE_STRENGTHS):
        noisy_df = _add_count_noise(period_df, strength, NOISE_SEED + number)
        noisy_data = make_windows(noisy_df, bundle["scaler"])
        noisy, score = _period_scores(
            noisy_data, model, bundle, CALIBRATION_END, None, score_function
        )
        if base_keys != list(zip(noisy["sensor"], noisy["time"])):
            raise ValueError("Rauschen hat die Zeitachse veraendert")
        noisy_alarm, _, _ = detector_output(
            score, data["sensor"], data["time"], observed, bundle["detector"]
        )
        additional = noisy_alarm & ~base_alarm
        result["strengths"][str(strength)] = {
            "seed": NOISE_SEED + number,
            "additional_false_alarm_count": int(additional[observed].sum()),
            "additional_false_alarm_rate": float(
                additional[observed].sum() / denominator
            ),
        }
    return result


def _detector_candidates(base):
    data, scores = base["data"], base["score"]
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


def select_detector(df, base_data, model, bundle, score_function=None):
    """Wählt die Alarmregel nur mit Kalibrierungsinjektionen."""
    base, runs = _prepare_injection_runs(
        df, base_data, model, bundle, TUNE_END, CALIBRATION_END,
        CALIBRATION_SEEDS, include_mild=False,
        score_function=score_function,
    )
    data = base["data"]
    observed = data["observed"]
    reports = []
    for detector in _detector_candidates(base):
        base_alarm, _, _ = detector_output(
            base["score"], data["sensor"], data["time"], observed, detector
        )
        rate = float(base_alarm[observed].mean())
        if rate > MAX_CALIBRATION_ALARM_RATE:
            continue
        by_type = {name: [] for name in ANOMALY_TYPES}
        for run in runs:
            event = _evaluate_injection_run(
                run, base, detector, base_alarm, include_point_metrics=False
            )["incremental_event"]
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


def _reference_detector(bundle):
    return {"point_threshold": bundle["calibrator"]["threshold"]}


def injection_benchmark(df, base_data, model, bundle, score_function=None):
    """Prüft die eingefrorene Alarmregel mit unabhängigen Testinjektionen."""
    base, runs = _prepare_injection_runs(
        df, base_data, model, bundle, CALIBRATION_END, None,
        TEST_SEEDS, include_mild=True, score_function=score_function,
    )
    results = _aggregate_injection_runs(runs, base, bundle["detector"])
    reference = _aggregate_injection_runs(
        runs, base, _reference_detector(bundle)
    )
    for anomaly_type in ANOMALY_TYPES:
        for strength in reference[anomaly_type]["strengths"].values():
            strength.pop("raw_score_ranking")
    results["noise_robustness"] = _noise_robustness(
        df, base, model, bundle, score_function
    )
    return results, reference
