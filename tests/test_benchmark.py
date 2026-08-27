"""Kleine Regressionstests fuer den synthetischen Benchmark."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import benchmark
from src.benchmark_evaluation import _add_count_noise
from src.forecaster.common import CALIBRATION_END
from src.benchmark_metrics import (
    _aggregate_injection_runs, _event_metrics, _ranking_metrics,
)


class MetricTests(unittest.TestCase):
    def test_missing_scores_rank_last_without_shrinking_the_population(self):
        result = _ranking_metrics(
            [True, False, False], [np.nan, 0.2, 0.1]
        )

        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["positive_count"], 1)
        self.assertEqual(result["missing_score_count"], 1)
        self.assertEqual(result["roc_auc"], 0.0)
        self.assertAlmostEqual(result["average_precision"], 1 / 3)

    def test_fragmented_prediction_matches_only_one_event(self):
        frame = pd.DataFrame({
            "sensor": "Sensor A",
            "time": pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC"),
            "label": [True, True, True],
            "prediction": [True, False, True],
        })

        result = _event_metrics(frame)

        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["mean_iou"], 1 / 3)

    def test_counts_and_score_curves_are_aggregated_separately(self):
        times = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
        base = {
            "data": {
                "sensor": np.repeat("Sensor A", 3),
                "time": times.to_numpy(dtype="datetime64[us]"),
                "observed": np.ones(3, dtype=bool),
            },
            "score": np.zeros(3),
        }
        common = {
            "anomaly_type": "spike",
            "strength": "1.0",
            "event_count": 1,
            "realized_strength": 1.0,
            "labels": np.array([True, False, False]),
        }
        runs = [
            {**common, "score": np.array([1.0, 0.0, 0.0])},
            {**common, "score": np.array([0.0, 1.0, 0.0])},
        ]
        result = _aggregate_injection_runs(
            runs, base, {"point_threshold": 0.5}
        )["spike"]["strengths"]["1.0"]

        point = result["incremental_point"]
        self.assertEqual(
            point["confusion_matrix"], {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
        )
        self.assertNotIn("roc_auc", point)
        ranking = result["raw_score_ranking"]
        self.assertIsNotNone(ranking["roc_auc"])
        self.assertLessEqual(len(ranking["roc_curve"]["false_positive_rate"]), 101)
        self.assertEqual(
            len(ranking["roc_curve"]["false_positive_rate"]),
            len(ranking["roc_curve"]["true_positive_rate"]),
        )


class NoiseTests(unittest.TestCase):
    def test_count_noise_is_reproducible_and_leaves_imputed_rows_untouched(self):
        frame = pd.DataFrame({
            "timestamp": [
                CALIBRATION_END - pd.Timedelta(days=1),
                CALIBRATION_END + pd.Timedelta(days=1),
                CALIBRATION_END + pd.Timedelta(days=2),
            ],
            "name": "Sensor A",
            "visitors": [100, 100, 100],
            "avgDuration": 5.0,
            "is_imputed": [False, False, True],
        })
        first = _add_count_noise(frame, 0.15, seed=7)
        second = _add_count_noise(frame, 0.15, seed=7)

        np.testing.assert_array_equal(first["visitors"], second["visitors"])
        self.assertEqual(first.loc[0, "visitors"], frame.loc[0, "visitors"])
        self.assertEqual(first.loc[2, "visitors"], frame.loc[2, "visitors"])
        self.assertGreaterEqual(first.loc[1, "visitors"], 0)
        self.assertEqual(first.loc[1, "visitors"] % 1, 0)


class BenchmarkBundleTests(unittest.TestCase):
    def test_existing_detector_is_reselected_and_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            bundle_path = model_dir / "model_bundle.json"
            bundle_path.write_text(json.dumps({
                "model": None, "scaler": {},
                "detector": {"point_threshold": 999},
            }), encoding="utf-8")
            new_detector = {"point_threshold": 1}
            with (
                patch.object(
                    benchmark, "_model_setup",
                    return_value=(model_dir, object()),
                ),
                patch.object(benchmark, "load_data", return_value=object()),
                patch.object(benchmark, "make_windows", return_value=object()),
                patch.object(benchmark, "split_masks", return_value=object()),
                patch.object(benchmark, "validate_bundle_data") as validate,
                patch.object(
                    benchmark, "select_detector",
                    return_value=(new_detector, {"selected": "new"}),
                ) as select,
                patch.object(
                    benchmark, "injection_benchmark", return_value=({}, {}),
                ),
                patch.object(benchmark, "_benchmark_rows", return_value=[]),
            ):
                benchmark._run_one("seasonal", Path("unused.json"))

            saved = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["detector"], new_detector)
            validate.assert_called_once()
            select.assert_called_once()


if __name__ == "__main__":
    unittest.main()
