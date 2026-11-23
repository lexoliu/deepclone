"""Causal batches must preserve incremental decoder outputs and KV history."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerDecoderLayer

from artifact import convert, enumerated_axis
from transformer import DecoderChunk


class Prefill(unittest.TestCase):
    def fixture(self, channels, heads, kv_heads, width, intermediate, context, count):
        config = Qwen3TTSTalkerConfig(
            hidden_size=channels,
            intermediate_size=intermediate,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=width,
            num_hidden_layers=1,
        )
        config._attn_implementation = "eager"
        with torch.random.fork_rng():
            torch.manual_seed(23)
            layers = [Qwen3TTSTalkerDecoderLayer(config, 0)]
        model = DecoderChunk(layers, config, context).eval().float()
        hidden = torch.sin(
            torch.arange(channels * count).reshape(1, channels, 1, count).float()
        )
        frequencies = config.rope_theta ** (-torch.arange(0, width, 2).float() / width)
        angles = frequencies[:, None] * torch.arange(count)[None, :]
        angles = torch.cat((angles, angles)).reshape(1, 1, width, count)
        cosine, sine = angles.cos(), angles.sin()

        def inputs(start, length):
            mask = torch.full((1, 1, length, context), torch.finfo(torch.float16).min)
            for index in range(length):
                mask[
                    ..., index, context - start - length : context - length + index + 1
                ] = 0
            return (
                hidden[..., start : start + length],
                cosine[..., start : start + length],
                sine[..., start : start + length],
                mask,
            )

        return model, inputs

    def test_batches_preserve_outputs_states_and_reset_at_context_boundary(self):
        context = 16
        model, inputs = self.fixture(32, 4, 2, 8, 64, context, context)
        with torch.inference_mode():
            expected = torch.cat(
                [model(*inputs(index, 1)) for index in range(context)], dim=-1
            )
            states = {name: value.clone() for name, value in model.named_buffers()}
            for _ in range(2):
                model.reset()
                offset = 0
                outputs = []
                for length in [4, 1, 8, 1, 1, 1]:
                    outputs.append(model(*inputs(offset, length)))
                    offset += length
                torch.testing.assert_close(torch.cat(outputs, dim=-1), expected)
                for name, value in model.named_buffers():
                    torch.testing.assert_close(value, states[name])

    def test_native_mixed_lengths_preserve_kv_state_and_reset(self):
        # Use the shipped cache layout to exercise the same native state path.
        count = 16
        model, inputs = self.fixture(1024, 16, 8, 128, 3072, 1024, count)
        model.half()
        names = ["hidden", "cosine", "sine", "mask"]
        examples = tuple(value.half() for value in inputs(0, 1))
        shapes = {
            name: enumerated_axis(value, (axis,), [1, 4, 8])
            for name, value, axis in zip(names, examples, [3, 3, 3, 2])
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "decoder.mlmodelc"
            convert(
                model,
                examples,
                names,
                ["hidden_out"],
                output,
                16,
                list(model.named_buffers()),
                shapes,
            )
            native = ct.models.CompiledMLModel(
                str(output), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            for _ in range(2):
                state = native.make_state()
                model.reset()
                offset = 0
                for length in [4, 1, 8, 1, 1, 1]:
                    batch = tuple(value.half() for value in inputs(offset, length))
                    with torch.inference_mode():
                        expected = model(*batch).float().numpy()
                    actual = native.predict(
                        dict(zip(names, [value.numpy() for value in batch])),
                        state=state,
                    )["hidden_out"].astype(np.float32)
                    self.assertLess(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected),
                        0.02,
                    )
                    for name, value in model.named_buffers():
                        expected_state = value.float().numpy()
                        actual_state = state.read_state(name).astype(np.float32)
                        self.assertLess(
                            np.linalg.norm(actual_state - expected_state)
                            / np.linalg.norm(expected_state),
                            0.02,
                            (
                                name,
                                offset,
                                length,
                                float(np.linalg.norm(actual_state)),
                                float(np.linalg.norm(expected_state)),
                            ),
                        )
                    offset += length
                self.assertEqual(offset, count)


if __name__ == "__main__":
    unittest.main()
