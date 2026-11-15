"""Export voice conditioning and waveform reconstruction models."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.mil.scope import ScopeSource
from qwen_tts import Qwen3TTSModel

from artifact import save
from audio_models import AudioDecoder, AudioEncoder


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
    parser.add_argument("--weight-bits", type=int, choices=[4, 8, 16], default=16)
    parser.add_argument("--compute-precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    if args.capacity <= 0:
        parser.error("capacity must be positive")
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    codec = reference.model.speech_tokenizer.model
    if args.component == "speaker":
        module = reference.model.speaker_encoder
        shape = (
            1,
            args.capacity,
            reference.model.config.speaker_encoder_config.mel_dim,
        )
        sample = torch.zeros(shape)
        input_shape = (
            1,
            ct.RangeDim(
                lower_bound=2, upper_bound=args.capacity, default=args.capacity
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
                lower_bound=1920, upper_bound=args.capacity, default=args.capacity
            ),
        )
        input_name, output_name = "audio", "codes"
        input_dtype, output_dtype = np.float32, np.int32
    else:
        module = AudioDecoder(codec.decoder, args.capacity)
        sample = torch.zeros(
            1, codec.decoder.config.num_quantizers, args.capacity, dtype=torch.int32
        )
        input_shape = tuple(sample.shape)
        input_name, output_name = "codes", "audio"
        input_dtype, output_dtype = np.int32, np.float32
    module.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, (sample,), check_trace=False)
    model = ct.convert(
        traced,
        minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name=input_name, shape=input_shape, dtype=input_dtype)],
        outputs=[ct.TensorType(name=output_name, dtype=output_dtype)],
        compute_precision=ct.precision.FLOAT32
        if args.compute_precision == "fp32"
        else ct.transform.FP16ComputePrecision(
            op_selector=lambda op: (
                "NearestCode"
                not in op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
            )
        ),
        skip_model_load=True,
    )
    save(model, args.output, args.weight_bits)
    args.output.with_suffix(".contract.json").write_text(
        json.dumps(
            {
                "component": args.component,
                "compute_precision": args.compute_precision,
                "weight_bits": args.weight_bits,
                "capacity": args.capacity,
                "sample_rate": codec.input_sample_rate,
                "samples_per_frame": codec.decode_upsample_rate,
                "code_groups": codec.encoder_valid_num_quantizers,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
