"""Validate dynamic causal masks through the Core ML conversion boundary."""

import unittest

import coremltools as ct
import numpy as np
import torch

from audio_models import causal_mask


class Mask(torch.nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def forward(self, scores):
        mask = causal_mask(scores.shape[-1], self.window)
        return torch.softmax(scores + mask, dim=-1)


class DynamicMask(unittest.TestCase):
    def test_attention_matches_at_multiple_lengths(self):
        for window in [None, 3]:
            module = Mask(window).eval()
            traced = torch.jit.trace(module, torch.zeros(1, 1, 7, 7))
            for precision in [ct.precision.FLOAT32, ct.precision.FLOAT16]:
                length = ct.RangeDim(lower_bound=1, upper_bound=11, default=7)
                model = ct.convert(
                    traced,
                    minimum_deployment_target=ct.target.macOS15,
                    inputs=[
                        ct.TensorType(
                            name="scores",
                            shape=(1, 1, length, length),
                            dtype=np.float32,
                        )
                    ],
                    outputs=[ct.TensorType(name="probabilities", dtype=np.float32)],
                    compute_precision=precision,
                    compute_units=ct.ComputeUnit.CPU_ONLY,
                )
                for size in [1, 3, 7, 11]:
                    scores = torch.sin(torch.arange(size * size).float()).reshape(
                        1, 1, size, size
                    )
                    expected = scores.clone()
                    for query in range(size):
                        for key in range(size):
                            if key > query or (
                                window is not None and query - key >= window
                            ):
                                expected[0, 0, query, key] = -torch.inf
                    expected = expected.softmax(dim=-1).numpy()
                    actual = model.predict({"scores": scores.numpy()})["probabilities"]
                    np.testing.assert_allclose(
                        actual, expected, rtol=0.002, atol=0.0002
                    )


if __name__ == "__main__":
    unittest.main()
