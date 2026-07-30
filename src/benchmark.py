"""Gemeinsamer synthetischer Vergleich für alle Anomaliemodelle.

Aufruf aus dem Projektordner: ``python -m src.benchmark``
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

from .benchmark_evaluation import (
    TEST_SEEDS, _forecast_scores, injection_benchmark, select_detector,
)
from .benchmark_injections import ANOMALY_TYPES
from .benchmark_metrics import _benchmark_rows
from .forecaster.common import (
    DATA_PATH, MODEL_DIR, load_data, make_windows,
)


def _mean_strong_curve(report, section, x_key, y_key):
    grid, values, aucs = np.linspace(0, 1, 101), [], []
    for anomaly_type in ANOMALY_TYPES:
        if anomaly_type not in report["results"]:
            continue
        result = report["results"][anomaly_type]
        strength = "6_hours" if anomaly_type == "stuck_at" else "2.0"
        ranking = result["strengths"][strength]["raw_score_ranking"]
        curve = ranking.get(section)
        if curve:
            x, y = np.asarray(curve[x_key]), np.asarray(curve[y_key])
            order = np.argsort(x)
            values.append(np.interp(grid, x[order], y[order]))
            aucs.append(ranking[
                "roc_auc" if section == "roc_curve" else "average_precision"
            ])
    return grid, np.mean(values, axis=0), float(np.mean(aucs))


def _save_ranking_curves(reports, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    specs = [
        ("roc_curve", "false_positive_rate", "true_positive_rate", "ROC"),
        ("pr_curve", "recall", "precision", "PR"),
    ]
    for axis, (section, x_key, y_key, title) in zip(axes, specs):
        for model, report in reports.items():
            x, y, auc = _mean_strong_curve(report, section, x_key, y_key)
            metric = "AUC" if section == "roc_curve" else "AP"
            axis.plot(x, y, label=f"{model} ({metric}={auc:.3f})")
        axis.set(xlabel=x_key, ylabel=y_key, title=f"{title}, starke Anomalien")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _model_setup(name):
    if name == "forecaster":
        return MODEL_DIR, _forecast_scores
    if name == "autoencoder":
        from .autoencoder.train import MODEL_DIR as directory
        from .autoencoder.train import reconstruction_scores
        return directory, reconstruction_scores
    if name == "seasonal":
        from .seasonal_baseline import MODEL_DIR as directory
        from .seasonal_baseline import seasonal_scores
        return directory, seasonal_scores
    raise ValueError(f"Unbekanntes Modell: {name}")


def _run_one(name, data_path):
    """Wendet exakt denselben Test auf ein gespeichertes Modell an."""
    model_dir, score_function = _model_setup(name)
    model_dir = Path(model_dir)
    with open(model_dir / "model_bundle.json", encoding="utf-8") as file:
        bundle = json.load(file)
    df = load_data(data_path)
    data = make_windows(df, bundle["scaler"])
    model = (
        tf.keras.models.load_model(model_dir / bundle["model"], compile=False)
        if bundle.get("model") else None
    )
    if "detector" not in bundle:
        detector, selection = select_detector(
            df, data, model, bundle, score_function
        )
        bundle["detector"] = detector
        bundle["detector_selection"] = selection
        with open(
            model_dir / "model_bundle.json", "w", encoding="utf-8"
        ) as file:
            json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)
    results, reference = injection_benchmark(
        df, data, model, bundle, score_function
    )
    report = {
        "design": (
            "Injektion ausschließlich in gemessene Besucherwerte vor Windowing "
            "und Prediction; drei von der Kalibrierung getrennte "
            "Zufallsstartwerte; bewertet "
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
    rows = _benchmark_rows(results, reference)
    pd.DataFrame(rows).to_csv(
        model_dir / "injection_benchmark.csv", index=False
    )
    print(f"Vergleichstest gespeichert: {model_dir / 'injection_benchmark.csv'}")
    return report, [dict(model=name, **row) for row in rows]


def run_benchmark(
    model_names=("forecaster", "autoencoder", "seasonal"), data_path=DATA_PATH,
):
    """Vergleicht alle Modelle mit denselben Injektionen und Metriken."""
    reports, rows = {}, []
    for name in model_names:
        reports[name], model_rows = _run_one(name, data_path)
        rows.extend(model_rows)

    output = Path(__file__).resolve().parents[1] / "saved_models" / "benchmark"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False)
    with open(output / "model_comparison.json", "w", encoding="utf-8") as file:
        json.dump(reports, file, ensure_ascii=False, indent=2, allow_nan=False)
    _save_ranking_curves(reports, output / "roc_pr_comparison.png")
    print(f"Modellvergleich gespeichert: {output / 'model_comparison.csv'}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--models", nargs="+", choices=["forecaster", "autoencoder", "seasonal"],
        default=["forecaster", "autoencoder", "seasonal"],
    )
    args = parser.parse_args()
    run_benchmark(args.models, args.data)


if __name__ == "__main__":
    main()
