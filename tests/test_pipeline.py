"""Kleine Regressionstests für die zwei wichtigsten Fehlerquellen."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecaster.common import anomaly_scores, fit_calibrator
from src.benchmark import _inject, _injection_events
from src.detector import detector_output
from src.autoencoder.train import add_masking_noise, sequences
from src.forecaster.evaluate import _evaluate_labels
from src.preprocessing import process_and_impute_full_data


class PreprocessingTests(unittest.TestCase):
    def _write_data(self, future_value):
        records = []
        for hour, value in [(0, 10), (3, future_value)]:
            records.append({
                "timestamp": f"2025-01-01T{hour:02d}:00:00Z",
                "name": "Sensor A",
                "visitors": value,
                "avgDuration": value / 2,
            })
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "raw.json"
        path.write_text(json.dumps({"sensordata": records}), encoding="utf-8")
        return directory, path

    def test_future_value_does_not_change_previous_imputation(self):
        temp_a, path_a = self._write_data(40)
        temp_b, path_b = self._write_data(400)
        try:
            a = process_and_impute_full_data(path_a, "2025-01-02T00:00:00Z")
            b = process_and_impute_full_data(path_b, "2025-01-02T00:00:00Z")
            vorher_a = a[a["timestamp"] < pd.Timestamp("2025-01-01T03:00:00Z")]
            vorher_b = b[b["timestamp"] < pd.Timestamp("2025-01-01T03:00:00Z")]
            np.testing.assert_array_equal(vorher_a["visitors"], vorher_b["visitors"])
            self.assertEqual(
                vorher_a["imputation_method"].tolist(),
                ["observed", "causal_ffill", "causal_ffill"],
            )
        finally:
            temp_a.cleanup()
            temp_b.cleanup()

    def test_systemwide_zero_is_treated_as_missing(self):
        records = []
        for sensor in range(10):
            for hour, value in [(0, 10), (1, 0), (2, 20)]:
                records.append({
                    "timestamp": f"2025-01-01T{hour:02d}:00:00Z",
                    "name": f"Sensor {sensor}",
                    "visitors": value,
                    "avgDuration": value / 2,
                })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps({"sensordata": records}), encoding="utf-8")
            result = process_and_impute_full_data(path, "2025-01-02T00:00:00Z")

        outage = result[result["timestamp"] == pd.Timestamp("2025-01-01T01:00:00Z")]
        self.assertTrue(outage["is_imputed"].all())
        self.assertTrue((outage["imputation_method"] == "causal_ffill").all())


class CalibrationTests(unittest.TestCase):
    def test_only_visitors_trigger_the_anomaly_score(self):
        prediction = np.zeros((200, 2, 3), dtype=float)
        prediction[..., 0] = -1
        prediction[..., 2] = 1
        actual = np.zeros((200, 2), dtype=float)
        actual[-3:] = 3
        calibrator = fit_calibrator(actual, prediction, alpha=0.02)
        _, normal_score, _ = anomaly_scores(actual[:1], prediction[:1], calibrator)
        extreme = np.array([[10.0, 0.0]])
        _, extreme_score, _ = anomaly_scores(extreme, prediction[:1], calibrator)
        duration_only = np.array([[0.0, 10.0]])
        _, duration_score, _ = anomaly_scores(
            duration_only, prediction[:1], calibrator
        )
        self.assertLessEqual(normal_score[0], calibrator["threshold"])
        self.assertGreater(extreme_score[0], calibrator["threshold"])
        self.assertEqual(duration_score[0], normal_score[0])


class InjectionTests(unittest.TestCase):
    def test_injection_changes_only_the_copy_and_produces_a_label(self):
        frame = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-07-10T00:00:00Z")],
            "name": ["Sensor A"], "visitors": [10.0], "avgDuration": [5.0],
            "is_imputed": [False], "imputation_method": ["observed"],
        })
        scaler = {"sensors": ["Sensor A"], "median": [[0.0, 0.0]],
                  "scale": [[1.0, 1.0]]}
        events = [{"indices": np.array([0]), "feature": "visitors",
                   "direction": 1}]
        injected, labels, realized = _inject(frame, events, "spike", 1.0, scaler)
        self.assertEqual(frame.loc[0, "visitors"], 10.0)
        self.assertGreater(injected.loc[0, "visitors"], 10.0)
        self.assertEqual(len(labels), 1)
        self.assertGreater(realized[0], 0.9)

    def test_all_injection_types_target_visitors(self):
        times = pd.date_range("2025-07-10", periods=6, freq="1h", tz="UTC")
        frame = pd.DataFrame({
            "timestamp": times,
            "name": "Sensor A",
            "visitors": np.arange(10, 16),
            "avgDuration": np.arange(5, 11),
            "is_imputed": False,
        })
        for anomaly_type in ["spike", "dip", "level_shift", "drift", "stuck_at"]:
            events = _injection_events(frame, anomaly_type, times[0], seed=42)
            self.assertTrue(events)
            self.assertEqual(events[0]["feature"], "visitors")


class DetectorTests(unittest.TestCase):
    def test_persistence_is_causal_and_resets_at_missing_target(self):
        scores = np.array([0.4, 0.4, -0.1, 0.4, np.nan,
                           0.4, 0.4, 0.4, 0.4])
        times = pd.date_range("2025-01-01", periods=len(scores), freq="1h")
        observed = np.array([True, True, True, True, False,
                             True, True, True, True])
        detector = {
            "point_threshold": 0.8,
            "low_threshold": 0.3,
            "persistence_window": 4,
            "required_exceedances": 3,
        }
        alarms, counts, _ = detector_output(
            scores, np.repeat("Sensor A", len(scores)), times,
            observed, detector,
        )
        np.testing.assert_array_equal(
            alarms, [False, False, False, True, False,
                     False, False, False, True],
        )
        self.assertTrue(np.isnan(counts[4]))
        self.assertTrue(np.isnan(counts[7]))


class AutoencoderTests(unittest.TestCase):
    def test_sequence_ends_with_current_target(self):
        data = {
            "history": np.zeros((2, 24, 8), dtype="float32"),
            "y_scaled": np.array([[1, 2], [3, 4]], dtype="float32"),
            "observed": np.array([True, False]),
        }
        result = sequences(data)
        self.assertEqual(result.shape, (2, 24, 3))
        np.testing.assert_array_equal(result[:, -1, :2], data["y_scaled"])
        np.testing.assert_array_equal(result[:, -1, 2], [1, 0])

    def test_masking_keeps_original_and_hides_measured_values(self):
        original = np.ones((3, 24, 3), dtype="float32")
        noisy = add_masking_noise(original)
        np.testing.assert_array_equal(original, np.ones_like(original))
        self.assertTrue((noisy[:, :, 2] == 0).any(axis=1).all())


class LabelTests(unittest.TestCase):
    def test_empty_template_rows_are_ignored(self):
        candidates = pd.DataFrame({
            "sensor": ["Sensor A", "Sensor A"],
            "time": pd.to_datetime([
                "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"
            ]),
            "detector_score": [1.0, -1.0],
            "anomaly": [True, False],
        })
        labels = pd.DataFrame({
            "sensor": ["Sensor A", "Sensor A"],
            "time": candidates["time"],
            "label": ["1", ""],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            labels.to_csv(path, index=False)
            result = _evaluate_labels(path, candidates)
        self.assertEqual(result["matched_rows"], 1)
        self.assertEqual(result["metrics"]["confusion_matrix"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
