"""Kleine Checks für Modellsuche, Kontextqualität und saisonale Baseline."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.forecaster.common import (
    AUXILIARY_FEATURES, CALIBRATION_END, FEATURES, TRAIN_END, TUNE_END,
    WINDOW_SIZE, calibration_context_quality, make_windows,
    model_inputs, select_by_validation_loss, set_seed, split_masks,
    validate_bundle_data,
)
from src.preprocessing import TRAIN_END as PREPROCESSING_TRAIN_END
from src.seasonal_baseline import seasonal_scores


class ModelSearchTests(unittest.TestCase):
    def test_seed_is_reproducible(self):
        set_seed(7)
        first = np.random.random(3)
        set_seed(7)
        np.testing.assert_array_equal(first, np.random.random(3))

    def test_successive_halving_warm_starts_survivors(self):
        losses = {
            "a": [0.50, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42],
            "b": [0.72, 0.68, 0.60, 0.52, 0.40, 0.30, 0.25, 0.24],
            "c": [0.62, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52],
            "d": [0.80, 0.78],
            "e": [0.90, 0.88],
            "f": [1.00, 0.98],
        }
        calls = {name: [] for name in losses}

        def run(params, model, initial_epoch, target_epoch, stop):
            model = object() if model is None else model
            calls[params["id"]].append(
                (id(model), initial_epoch, target_epoch, stop)
            )
            values = losses[params["id"]][initial_epoch:target_epoch]
            history = SimpleNamespace(history={
                "loss": [value + 0.1 for value in values],
                "val_loss": values,
            })
            return history, model, 10

        selected, trials = select_by_validation_loss([
            {"id": name} for name in losses
        ], run, initial_epochs=2, max_epochs=8, reduction_factor=2)
        self.assertEqual(selected["params"]["id"], "b")
        self.assertEqual(selected["best_epoch"], 8)
        self.assertEqual(selected["epochs_run"], 8)
        self.assertEqual(selected["history"]["val_loss"], losses["b"])
        self.assertEqual(
            selected["history"]["loss"],
            [value + 0.1 for value in losses["b"]],
        )
        self.assertEqual(len(trials), 6)
        self.assertTrue(all(call[1:] == (0, 2, False)
                            for call in (calls[name][0] for name in losses)))
        self.assertEqual({name for name, calls_ in calls.items() if len(calls_) > 1},
                         {"a", "b", "c"})
        self.assertEqual({name for name, calls_ in calls.items()
                          if calls_[-1][3]}, {"a", "b"})
        for name in ["a", "b"]:
            self.assertEqual(
                [call[1:] for call in calls[name]],
                [(0, 2, False), (2, 4, False), (4, 8, True)],
            )
            self.assertEqual(len({call[0] for call in calls[name]}), 1)
        self.assertEqual(
            {trial["params"]["id"]: trial["status"] for trial in trials},
            {name: "selected" if name == "b" else "pruned" for name in losses},
        )
        trial_by_id = {trial["params"]["id"]: trial for trial in trials}
        self.assertEqual(trial_by_id["a"]["pruned_after_round"], 3)
        self.assertNotIn("history", trial_by_id["b"])

    def test_single_candidate_runs_once_from_scratch_with_early_stopping(self):
        calls = []

        def run(params, model, initial_epoch, target_epoch, stop):
            calls.append((model, initial_epoch, target_epoch, stop))
            history = SimpleNamespace(history={
                "loss": [0.4, 0.3, 0.2],
                "val_loss": [0.5, 0.35, 0.25],
            })
            return history, object(), 10

        selected, trials = select_by_validation_loss(
            [{"id": "only"}], run, initial_epochs=2, max_epochs=5,
        )

        self.assertEqual(calls, [(None, 0, 5, True)])
        self.assertEqual(selected["best_epoch"], 3)
        self.assertEqual(selected["epochs_run"], 3)
        self.assertEqual(len(trials), 1)

    def test_calibration_quality_warns_for_imputed_majority(self):
        data = {"imputed_fraction": np.array([0.75, 0.8, 0.1])}
        result = calibration_context_quality(data, np.ones(3, dtype=bool))
        self.assertEqual(result["more_than_half_imputed_count"], 2)
        self.assertIsNotNone(result["warning"])

    def test_shared_chronological_split_boundaries(self):
        self.assertEqual(PREPROCESSING_TRAIN_END, TRAIN_END)
        data = {
            "time": np.array([
                (TRAIN_END - pd.Timedelta(hours=1)).to_datetime64(),
                TRAIN_END.to_datetime64(),
                TUNE_END.to_datetime64(),
                CALIBRATION_END.to_datetime64(),
            ]),
            "observed": np.ones(4, dtype=bool),
        }
        masks = split_masks(data)
        np.testing.assert_array_equal(masks["train"], [True, False, False, False])
        np.testing.assert_array_equal(masks["tune"], [False, True, False, False])
        np.testing.assert_array_equal(
            masks["calibration"], [False, False, True, False]
        )
        np.testing.assert_array_equal(masks["test"], [False, False, False, True])

    def test_bundle_must_match_current_features_splits_period_and_counts(self):
        data = {
            "time": np.array([
                (TRAIN_END - pd.Timedelta(hours=1)).to_datetime64(),
                TRAIN_END.to_datetime64(),
                TUNE_END.to_datetime64(),
                CALIBRATION_END.to_datetime64(),
            ]),
            "observed": np.ones(4, dtype=bool),
        }
        masks = split_masks(data)
        bundle = {
            "features": FEATURES,
            "window_size": WINDOW_SIZE,
            "auxiliary_features": AUXILIARY_FEATURES,
            "splits": {
                "train_end": str(TRAIN_END), "tune_end": str(TUNE_END),
                "calibration_end": str(CALIBRATION_END),
            },
            "data_period": {"last_target": str(CALIBRATION_END)},
            "sample_counts": {
                name: int(mask.sum()) for name, mask in masks.items()
            },
        }
        self.assertIsNone(
            validate_bundle_data(bundle, data, masks, require_auxiliary=True)
        )

        cases = [
            ("features", lambda value: value.update(features=["visitors"])),
            ("window_size", lambda value: value.update(window_size=12)),
            ("auxiliary_features", lambda value: value.update(auxiliary_features=[])),
            ("splits.train_end", lambda value: value["splits"].update(
                train_end=str(TUNE_END)
            )),
            ("last_target", lambda value: value["data_period"].update(
                last_target=str(TUNE_END)
            )),
            ("sample_counts", lambda value: value.update(sample_counts={})),
        ]
        for message, mutate in cases:
            with self.subTest(message=message):
                incompatible = deepcopy(bundle)
                mutate(incompatible)
                with self.assertRaisesRegex(ValueError, message):
                    validate_bundle_data(
                        incompatible, data, masks, require_auxiliary=True
                    )

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

    def test_weekly_and_summary_features_use_only_past_values(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range(
                "2025-01-01", periods=169, freq="1h", tz="UTC"
            ),
            "name": "Sensor A",
            "visitors": np.arange(169, dtype=float),
            "avgDuration": np.arange(169, dtype=float) + 1,
            "is_imputed": False,
        })
        scaler = {"sensors": ["Sensor A"], "median": [[0, 0]],
                  "scale": [[1, 1]]}
        data = make_windows(frame, scaler)
        context = data["future"][-1]
        scaled = np.log1p(frame[["visitors", "avgDuration"]].to_numpy())

        self.assertEqual(len(context), len(AUXILIARY_FEATURES))
        np.testing.assert_allclose(context[5:7], scaled[0])
        self.assertEqual(context[7], 1)
        np.testing.assert_allclose(
            context[8:10], scaled[144:168].mean(axis=0), atol=1e-6
        )
        np.testing.assert_allclose(
            context[10:12], scaled[144:168].std(axis=0), atol=1e-6
        )

    def test_imputed_weekly_reference_keeps_values_and_clears_its_mask(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range(
                "2025-01-01", periods=169, freq="1h", tz="UTC"
            ),
            "name": "Sensor A",
            "visitors": np.arange(169, dtype=float),
            "avgDuration": np.arange(169, dtype=float) + 1,
            "is_imputed": [True] + [False] * 168,
        })
        scaler = {
            "sensors": ["Sensor A"], "median": [[0, 0]], "scale": [[1, 1]],
        }
        data = make_windows(frame, scaler)
        future = model_inputs(data)["future"][-1]

        np.testing.assert_allclose(future[5:7], np.log1p([0, 1]))
        self.assertEqual(future[7], 0)


if __name__ == "__main__":
    unittest.main()
