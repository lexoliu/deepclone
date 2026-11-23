"""Export voice conditioning and waveform reconstruction models."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.mil.scope import ScopeSource
from qwen_tts import Qwen3TTSModel

from artifact import enumerated_axis, save
from audio_models import AudioEncoder
from streaming_audio import StreamingAudioDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component", choices=["speaker", "encoder", "decoder"], required=True
    )
    parser.add_argument(
        "--capacity",
        type=int,
        required=True,
        help="Mel frames, audio samples, or codec frames",
    )
    parser.add_argument("--weight-bits", type=int, choices=[4, 8, 16, 32])
    parser.add_argument("--batch-frames", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--compute-precision",
        choices=["fp16", "fp32"],
        help="Defaults to fp32 for reference conditioning and fp16 for waveform decoding",
    )
    args = parser.parse_args()
    if args.capacity <= 0:
        parser.error("capacity must be positive")
    if args.batch_frames != sorted(set(args.batch_frames)) or args.batch_frames[0] != 1:
        parser.error("batch-frames must increase from one")
    if args.batch_frames[-1] > args.capacity or (
        len(args.batch_frames) > 1 and args.component != "decoder"
    ):
        parser.error("batch-frames must fit the waveform decoder capacity")
    compute_precision = args.compute_precision or (
        "fp16" if args.component == "decoder" else "fp32"
    )
    precision_bits = 32 if compute_precision == "fp32" else 16
    weight_bits = args.weight_bits or precision_bits
    if weight_bits not in (4, 8, precision_bits):
        parser.error("uncompressed weight width must match compute precision")
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    codec = reference.model.speech_tokenizer.model
    if args.component == "speaker":
        module = reference.model.speaker_encoder
        # Reflection padding must be smaller than the time axis. Every temporal
        # convolution in this speaker encoder preserves length with SAME padding.
        minimum_frames = 1 + max(
            ((layer.kernel_size[0] - 1) * layer.dilation[0] + 1) // 2
            for layer in module.modules()
            if isinstance(layer, torch.nn.Conv1d)
            and layer.padding == "same"
            and layer.padding_mode == "reflect"
        )
        if args.capacity < minimum_frames:
            parser.error("speaker capacity is shorter than its reflection padding")
        shape = (
            1,
            args.capacity,
            reference.model.config.speaker_encoder_config.mel_dim,
        )
        sample = torch.zeros(shape)
        input_shape = (
            1,
            ct.RangeDim(
                lower_bound=minimum_frames,
                upper_bound=args.capacity,
                default=args.capacity,
            ),
            shape[-1],
        )
        input_name, output_name = "mel", "speaker"
        input_dtype, output_dtype = np.float32, np.float16
    elif args.component == "encoder":
        module = AudioEncoder(codec, args.capacity)
        sample = torch.zeros(1, 1, args.capacity)
        input_shape = (
            1,
            1,
            ct.RangeDim(
                lower_bound=codec.encode_downsample_rate,
                upper_bound=args.capacity,
                default=args.capacity,
            ),
        )
        input_name, output_name = "audio", "codes"
        input_dtype, output_dtype = np.float32, np.int32
    else:
        if compute_precision != "fp16" or weight_bits != 16:
            parser.error("stateful waveform decoding requires uncompressed fp16")
        module = StreamingAudioDecoder(codec.decoder)
        sample = torch.zeros(
            1, codec.decoder.config.num_quantizers, 1, dtype=torch.int32
        )
        input_shape = enumerated_axis(sample, (2,), args.batch_frames)
        input_name, output_name = "codes", "audio"
        input_dtype, output_dtype = np.int32, np.float32
    module.eval()
    samples = (sample,)
    inputs = [ct.TensorType(name=input_name, shape=input_shape, dtype=input_dtype)]
    states = []
    if args.component == "decoder":
        config = codec.decoder.config
        cosine = torch.ones(1, 1, 1, config.head_dim)
        sine = torch.zeros_like(cosine)
        mask = torch.full(
            (1, 1, 1, config.sliding_window), torch.finfo(torch.float16).min
        )
        mask[..., -1] = 0
        samples = (sample, cosine, sine, mask)
        inputs.extend(
            ct.TensorType(
                name=name,
                shape=enumerated_axis(value, axes, args.batch_frames),
                dtype=np.float16,
            )
            for name, value, axes in zip(
                ["cosine", "sine", "mask"], samples[1:], [(2,), (2,), (2, 3)]
            )
        )
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(value.shape), dtype=np.float16),
                name=name,
            )
            for name, value in module.named_buffers()
        ]
    with torch.inference_mode():
        traced = torch.jit.trace(module, samples, check_trace=False)
    if states:
        module.reset()
    model = ct.convert(
        traced,
        minimum_deployment_target=ct.target.macOS15,
        inputs=inputs,
        outputs=[ct.TensorType(name=output_name, dtype=output_dtype)],
        states=states,
        compute_precision=ct.precision.FLOAT32
        if compute_precision == "fp32"
        else ct.transform.FP16ComputePrecision(
            op_selector=lambda op: (
                "NearestCode"
                not in op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
            )
        ),
        skip_model_load=True,
    )
    save(model, args.output, weight_bits)
    args.output.with_suffix(".contract.json").write_text(
        json.dumps(
            {
                "component": args.component,
                "compute_precision": compute_precision,
                "weight_bits": weight_bits,
                "capacity": args.capacity,
                "sample_rate": codec.input_sample_rate,
                "samples_per_frame": codec.decode_upsample_rate,
                "code_groups": codec.encoder_valid_num_quantizers,
                **(
                    {
                        "execution": "stateful_frame",
                        "batch_frames": args.batch_frames,
                        "attention_window": codec.decoder.config.sliding_window,
                        "rotary_frequencies": codec.decoder.pre_transformer.rotary_emb.inv_freq.tolist(),
                        "rotary_scaling": codec.decoder.pre_transformer.rotary_emb.attention_scaling,
                    }
                    if args.component == "decoder"
                    else {}
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
