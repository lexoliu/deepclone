"""Convert an exact Qwen decoder layer group with native Core ML KV state."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from artifact import save
from transformer import DecoderChunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder", choices=["talker", "predictor"], default="talker")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--weight-bits", type=int, choices=[4, 8, 16], default=16)
    args = parser.parse_args()
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    decoder = reference.model.talker
    if args.decoder == "predictor":
        decoder = decoder.code_predictor
    config = decoder.config
    layers = decoder.model.layers[args.start : args.start + args.layers]
    if len(layers) != args.layers:
        raise ValueError("Requested layer range lies outside the model")
    chunk = DecoderChunk(layers, config, args.context).eval().half()
    shapes = [
        (1, config.hidden_size, 1, 1),
        (1, 1, config.head_dim, 1),
        (1, 1, config.head_dim, 1),
        (1, 1, 1, args.context),
    ]
    inputs = tuple(torch.zeros(shape, dtype=torch.float16) for shape in shapes)
    inputs[1].fill_(1)
    inputs[3].fill_(torch.finfo(torch.float16).min)
    inputs[3][..., -1] = 0
    with torch.inference_mode():
        traced = torch.jit.trace(chunk, inputs, check_trace=False)
    chunk.reset()
    names = ["hidden", "cosine", "sine", "mask"]
    model = ct.convert(
        traced,
        minimum_deployment_target=ct.target.macOS15,
        inputs=[
            ct.TensorType(name=name, shape=shape, dtype=np.float16)
            for name, shape in zip(names, shapes)
        ],
        outputs=[ct.TensorType(name="hidden_out", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(value.shape), dtype=np.float16),
                name=name,
            )
            for name, value in chunk.named_buffers()
        ],
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    save(model, args.output, args.weight_bits)


if __name__ == "__main__":
    main()
