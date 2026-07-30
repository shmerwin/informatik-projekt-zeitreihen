"""Kleine Checks für Modellsuche, Kontextqualität und saisonale Baseline."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.forecaster.common import (
    calibration_context_quality, make_windows, select_by_validation_loss,
)
from src.seasonal_baseline import seasonal_scores


class ModelSearchTests(unittest.TestCase):
    def test_minimum_tune_loss_wins(self):
        losses = {"a": [0.5, 0.4], "b": [0.6, 0.2, 0.3], "c": [0.7]}

        def run(params):
            return SimpleNamespace(history={"val_loss": losses[params["id"]]}), 10

        selected, trials = select_by_validation_loss(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}], run
        )
        self.assertEqual(selected["params"]["id"], "b")
        self.assertEqual(selected["best_epoch"], 2)
        self.assertEqual(len(trials), 3)

    def test_calibration_quality_warns_for_imputed_majority(self):
        data = {"imputed_fraction": np.array([0.75, 0.8, 0.1])}
        result = calibration_context_quality(data, np.ones(3, dtype=bool))
        self.assertEqual(result["more_than_half_imputed_count"], 2)
        self.assertIsNotNone(result["warning"])

    def test_seasonal_score_requires_measured_t_minus_24(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=25, freq="1h", tz="UTC"),
            "name": "Sensor A",
            "visitors": np.arange(25),
            "avgDuration": np.arange(25),
            "is_imputed": [True] + [False] * 24,
        })
        scaler = {"sensors": ["Sensor A"], "median": [[0, 0]], "scale": [[1, 1]]}
        data = make_windows(frame, scaler)
        bundle = {
            "scaler": scaler,
            "residual_center": [0, 0],
            "residual_scale": [1, 1],
            "anomaly_feature": "visitors",
        }
        scores = seasonal_scores(data, None, bundle)
        self.assertFalse(data["seasonal_observed"][0])
        self.assertTrue(data["observed"][0])
        self.assertTrue(np.isnan(scores[0]))


if __name__ == "__main__":
    unittest.main()
