"""Kleine Regressionstests für die wichtigsten Pipeline-Regeln."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecaster.benchmark import classification_metrics, inject
from src.forecaster.evaluate import apply_detector
from src.forecaster.train import make_windows
from src.preprocessing import process_and_impute_full_data


class PreprocessingTests(unittest.TestCase):
    def _write(self, records, directory):
        path = Path(directory) / "raw.json"
        path.write_text(json.dumps({"sensordata": records}), encoding="utf-8")
        return path

    def test_future_value_does_not_change_previous_imputation(self):
        def records(future):
            return [
                {"timestamp": "2025-01-01T00:00:00Z", "name": "A",
                 "visitors": 10, "avgDuration": 5},
                {"timestamp": "2025-01-01T03:00:00Z", "name": "A",
                 "visitors": future, "avgDuration": future / 2},
            ]

        with (tempfile.TemporaryDirectory() as dir_a,
              tempfile.TemporaryDirectory() as dir_b):
            path_a = self._write(records(40), dir_a)
            path_b = self._write(records(400), dir_b)
            result_a = process_and_impute_full_data(path_a, "2025-01-02T00:00:00Z")
            result_b = process_and_impute_full_data(path_b, "2025-01-02T00:00:00Z")
            np.testing.assert_array_equal(
                result_a.iloc[:3]["visitors"], result_b.iloc[:3]["visitors"]
            )

    def test_systemwide_zero_is_treated_as_missing(self):
        records = []
        for sensor in range(10):
            for hour, value in [(0, 10 + sensor), (1, 0), (2, 12 + sensor)]:
                records.append({
                    "timestamp": f"2025-01-01T{hour:02d}:00:00Z",
                    "name": f"Sensor {sensor}",
                    "visitors": value,
                    "avgDuration": 0 if hour == 1 else 5,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(records, directory)
            result = process_and_impute_full_data(path, "2025-01-02T00:00:00Z")
            outage = result[result["timestamp"] == pd.Timestamp("2025-01-01T01:00:00Z")]
            self.assertEqual(len(outage), 10)
            self.assertTrue(outage["is_imputed"].all())
            self.assertTrue((outage["source_record_count"] == 1).all())


class WindowTests(unittest.TestCase):
    def test_imputation_flag_is_context_channel_and_target_status(self):
        times = pd.date_range("2025-01-01", periods=30, freq="1h", tz="UTC")
        frame = pd.DataFrame({
            "timestamp": times,
            "name": "Sensor A",
            "visitors": np.arange(30) + 1,
            "avgDuration": np.arange(30) / 2 + 1,
            "is_imputed": False,
        })
        frame.loc[23, "is_imputed"] = True
        frame.loc[25, "is_imputed"] = True
        data = make_windows(frame, {"mean": [0, 0], "scale": [1, 1]})
        self.assertEqual(data["X"].shape, (6, 24, 3))
        self.assertEqual(data["X"][0, -1, 2], 0)
        self.assertFalse(data["observed"][1])


class MetricTests(unittest.TestCase):
    def test_required_classification_metrics(self):
        labels = np.array([0, 0, 1, 1])
        alarms = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.7, 0.4, 0.9])
        result = classification_metrics(labels, alarms, scores)
        self.assertEqual(result["confusion_matrix"], {
            "tn": 1, "fp": 1, "fn": 1, "tp": 1,
        })
        self.assertAlmostEqual(result["temporal_iou"], 1 / 3)
        self.assertIn("roc_auc", result)

    def test_feature_thresholds_are_applied_separately(self):
        scores = np.array([
            [3.0, 7.0], [4.0, 2.0], [3.0, 6.0], [9.0, 9.0],
        ])
        observed = np.array([1, 1, 1, 0], dtype=bool)
        bundle = {
            "detector": {"feature_thresholds": [3.5, 6.5]},
        }
        alarm, feature_alarm = apply_detector(scores, observed, bundle)
        np.testing.assert_array_equal(alarm, [True, True, False, False])
        np.testing.assert_array_equal(feature_alarm[0], [False, True])
        np.testing.assert_array_equal(feature_alarm[1], [True, False])

class InjectionTests(unittest.TestCase):
    def test_injection_changes_only_copy(self):
        frame = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-07-10T00:00:00Z")],
            "name": ["A"], "visitors": [10.0], "avgDuration": [5.0],
            "is_imputed": [False],
        })
        event = [{"indices": np.array([0]), "feature": "visitors", "direction": 1}]
        changed, labels, realized = inject(
            frame, event, "spike", 1.0, {"mean": [0, 0], "scale": [1, 1]}
        )
        self.assertEqual(frame.loc[0, "visitors"], 10)
        self.assertGreater(changed.loc[0, "visitors"], 10)
        self.assertEqual(len(labels), 1)
        self.assertGreater(realized[0], 0.5)


if __name__ == "__main__":
    unittest.main()
