"""Exercise graph identities independently of model downloads."""

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from transformer import Projection, RmsNorm, gated_silu


class Equations(unittest.TestCase):
    def test_projection_preserves_bias_and_channel_order(self):
        weight = torch.sin(torch.arange(128 * 256).reshape(128, 256).float()) / 32
        bias = torch.cos(torch.arange(128).float())
        linear = SimpleNamespace(weight=weight, bias=bias)
        x = torch.sin(torch.arange(2 * 256 * 3).reshape(2, 256, 1, 3).float())
        expected = (
            F.linear(x.squeeze(2).transpose(1, 2), weight, bias)
            .transpose(1, 2)
            .unsqueeze(2)
        )
        torch.testing.assert_close(
            Projection(linear)(x), expected, rtol=1e-4, atol=1e-5
        )

    def test_scaled_norm_preserves_epsilon_across_magnitudes(self):
        norm = SimpleNamespace(
            weight=nn.Parameter(torch.ones(128)), variance_epsilon=1e-6
        )
        for exponent in range(-5, 5):
            x = torch.linspace(-1, 1, 128).reshape(1, 128, 1, 1) * math.pow(
                10, exponent
            )
            expected = x * torch.rsqrt(
                (x * x).mean(dim=1, keepdim=True) + norm.variance_epsilon
            )
            torch.testing.assert_close(RmsNorm(norm)(x), expected)
        x = torch.full((1, 128, 1, 1), 1000.0, dtype=torch.float16)
        actual = RmsNorm(norm).half()(x)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual, torch.ones_like(actual), atol=0.002, rtol=0.002
        )

    def test_gate_regrouping_matches_silu(self):
        gate = torch.linspace(-20, 20, 1024)
        up = torch.cos(gate)
        torch.testing.assert_close(gated_silu(gate, up), F.silu(gate) * up)


if __name__ == "__main__":
    unittest.main()
