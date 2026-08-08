"""Kleiner Vertragstest für den Transformer."""

import unittest

import numpy as np

from src.transformer.train import build_model


class TransformerTests(unittest.TestCase):
    def test_named_inputs_produce_ordered_quantiles(self):
        params = {
            "d_model": 24, "heads": 4, "blocks": 1, "ff_dim": 48,
            "dense": 24, "embedding": 4, "dropout": 0.1, "lr": 1e-3,
        }
        model = build_model(3, 8, params)
        output = model({
            "history": np.zeros((2, 24, 8), dtype="float32"),
            "future": np.zeros((2, 5), dtype="float32"),
            "sensor_id": np.zeros((2, 1), dtype="int32"),
        }).numpy()

        self.assertEqual(output.shape, (2, 2, 3))
        self.assertTrue(np.all(output[..., 0] <= output[..., 1]))
        self.assertTrue(np.all(output[..., 1] <= output[..., 2]))


if __name__ == "__main__":
    unittest.main()
