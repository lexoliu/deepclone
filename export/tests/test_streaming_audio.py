"""Exercise decoder histories, sliding attention and native state updates."""

import unittest

import coremltools as ct
import numpy as np
import torch
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    EuclideanCodebook,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderConfig,
)

from audio_models import AudioDecoder
from streaming_audio import StreamingAudioDecoder
from streaming_audio import metadata as rotary_metadata


def decoder():
    config = Qwen3TTSTokenizerV2DecoderConfig(
        latent_dim=8,
        codebook_dim=8,
        codebook_size=16,
        decoder_dim=32,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        num_quantizers=4,
        num_semantic_quantizers=1,
        sliding_window=5,
        upsample_rates=[2, 2],
        upsampling_ratios=[2],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = Qwen3TTSTokenizerV2Decoder(config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, EuclideanCodebook):
                module.embedding_sum.normal_()
                module.cluster_usage.uniform_(0.5, 2)
    return model


def metadata(model, position, dtype):
    rotary = model.pre_transformer.rotary_emb
    return rotary_metadata(
        rotary.inv_freq,
        rotary.attention_scaling,
        model.config.sliding_window,
        position,
        dtype,
    )


class StreamingAudio(unittest.TestCase):
    def test_cached_equations_preserve_full_decoder_and_reset(self):
        original = decoder()
        streaming = StreamingAudioDecoder(original).eval()
        codes = torch.arange(4 * 17, dtype=torch.int32).reshape(1, 4, 17) % 16
        with torch.inference_mode():
            expected = AudioDecoder(original, 17)(codes)
            for _ in range(2):
                streaming.reset()
                parts = [
                    streaming(
                        codes[..., index : index + 1],
                        *metadata(original, index, torch.float32),
                    )
                    for index in range(codes.shape[-1])
                ]
                torch.testing.assert_close(
                    torch.cat(parts, dim=-1), expected, atol=1e-6, rtol=1e-4
                )
            # A new official block must restart every convolution and KV state.
            streaming.reset()
            short = codes[..., -3:]
            expected = AudioDecoder(original, 3)(short)
            actual = torch.cat(
                [
                    streaming(
                        short[..., index : index + 1],
                        *metadata(original, index, torch.float32),
                    )
                    for index in range(3)
                ],
                dim=-1,
            )
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)

    def test_native_state_updates_and_independent_sequences(self):
        original = decoder()
        streaming = StreamingAudioDecoder(original).eval()
        codes = torch.arange(4 * 9, dtype=torch.int32).reshape(1, 4, 9) % 16
        sample = (codes[..., :1], *metadata(original, 0, torch.float32))
        with torch.inference_mode():
            traced = torch.jit.trace(streaming, sample, check_trace=False)
            streaming.reset()
            model = ct.convert(
                traced,
                minimum_deployment_target=ct.target.macOS15,
                inputs=[
                    ct.TensorType(name=name, shape=tuple(value.shape), dtype=dtype)
                    for name, value, dtype in zip(
                        ["codes", "cosine", "sine", "mask"],
                        sample,
                        [np.int32, np.float16, np.float16, np.float16],
                    )
                ],
                states=[
                    ct.StateType(
                        name=name,
                        wrapped_type=ct.TensorType(
                            shape=tuple(value.shape), dtype=np.float16
                        ),
                    )
                    for name, value in streaming.named_buffers()
                ],
                outputs=[ct.TensorType(name="audio", dtype=np.float32)],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            for units in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]:
                native = ct.models.MLModel(
                    model.get_spec(), weights_dir=model.weights_dir, compute_units=units
                )
                streaming.reset()
                first, second = native.make_state(), native.make_state()
                for index in range(codes.shape[-1]):
                    values = (
                        codes[..., index : index + 1],
                        *metadata(original, index, torch.float32),
                    )
                    inputs = dict(
                        zip(
                            ["codes", "cosine", "sine", "mask"],
                            [value.numpy() for value in values],
                        )
                    )
                    expected = streaming(*values).float().numpy()
                    actual = native.predict(inputs, state=first)["audio"]
                    independent = native.predict(inputs, state=second)["audio"]
                    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.02)
                    np.testing.assert_allclose(
                        actual, independent, atol=1e-6, rtol=1e-5
                    )


if __name__ == "__main__":
    unittest.main()
