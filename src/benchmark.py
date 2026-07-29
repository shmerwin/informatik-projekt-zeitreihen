"""Gemeinsamer synthetischer Vergleich für alle Anomaliemodelle.

Aufruf aus dem Projektordner: ``python -m src.benchmark``
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .benchmark_evaluation import (
    CALIBRATION_SEEDS, HIGH_ALPHAS, LOW_ALPHAS, MAX_CALIBRATION_ALARM_RATE,
    PERSISTENCE_RULES, TEST_SEEDS, _forecast_scores, injection_benchmark,
    select_detector,
)
from .benchmark_injections import ANOMALY_TYPES, _inject, _injection_events
from .benchmark_metrics import _benchmark_rows
from .forecaster.common import (
    ANOMALY_FEATURE, CALIBRATION_END, DATA_PATH, FEATURES, MODEL_DIR, TUNE_END,
    load_data, make_windows,
)


def _model_setup(name):
    if name == "forecaster":
        return MODEL_DIR, _forecast_scores
    if name == "autoencoder":
        from .autoencoder.train import MODEL_DIR as directory
        from .autoencoder.train import reconstruction_scores
        return directory, reconstruction_scores
    raise ValueError(f"Unbekanntes Modell: {name}")


def _run_one(name, data_path):
    """Wendet exakt denselben Test auf ein gespeichertes Modell an."""
    model_dir, score_function = _model_setup(name)
    model_dir = Path(model_dir)
    with open(model_dir / "model_bundle.json", encoding="utf-8") as file:
        bundle = json.load(file)
    df = load_data(data_path)
    data = make_windows(df, bundle["scaler"])
    model = tf.keras.models.load_model(
        model_dir / bundle["model"], compile=False
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
    pd.DataFrame(_benchmark_rows(results, reference)).to_csv(
        model_dir / "injection_benchmark.csv", index=False
    )
    print(f"Vergleichstest gespeichert: {model_dir / 'injection_benchmark.csv'}")
    return report, [dict(model=name, **row) for row in _benchmark_rows(
        results, reference
    )]


def run_benchmark(model_names=("forecaster", "autoencoder"), data_path=DATA_PATH):
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
    print(f"Modellvergleich gespeichert: {output / 'model_comparison.csv'}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--models", nargs="+", choices=["forecaster", "autoencoder"],
        default=["forecaster", "autoencoder"],
    )
    args = parser.parse_args()
    run_benchmark(args.models, args.data)


if __name__ == "__main__":
    main()
