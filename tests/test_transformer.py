"""Kleiner Vertragstest für den Transformer."""

import unittest

import numpy as np

from src.transformer.train import CANDIDATES, build_model


class TransformerTests(unittest.TestCase):
    def test_all_candidates_produce_ordered_quantiles(self):
        parameters = []
        inputs = {
            "history": np.zeros((2, 24, 8), dtype="float32"),
            "future": np.zeros((2, 5), dtype="float32"),
            "sensor_id": np.zeros((2, 1), dtype="int32"),
        }
        for params in CANDIDATES:
            with self.subTest(params=params):
                self.assertEqual(params["d_model"] % params["heads"], 0)
                model = build_model(3, 8, params)
                output = model(inputs).numpy()
                parameters.append(model.count_params())
                self.assertEqual(output.shape, (2, 2, 3))
                self.assertTrue(np.all(output[..., 0] <= output[..., 1]))
                self.assertTrue(np.all(output[..., 1] <= output[..., 2]))
        self.assertGreater(max(parameters), 10 * min(parameters))


if __name__ == "__main__":
    unittest.main()
