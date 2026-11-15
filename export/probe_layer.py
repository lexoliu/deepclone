"""Localize first-token conversion error without recurrent state."""

import json

import coremltools as ct
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from torch import nn

from transformer import DecoderLayer, gated_silu


class Probe(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        layer = self.layer
        normalized = layer.input_norm(x)
        value = layer.value(normalized)
        expanded = value.reshape(
            1, layer.kv_heads, layer.head_dim, 1
        ).repeat_interleave(layer.groups, dim=1)
        projected = layer.output(
            expanded.reshape(1, layer.heads * layer.head_dim, 1, 1)
        )
        residual = x + projected
        post = layer.post_norm(residual)
        gate, up = layer.gate(post), layer.up(post)
        activated = gated_silu(gate, up)
        down = layer.down(activated)
        return (
            normalized,
            value,
            projected,
            residual,
            post,
            gate,
            up,
            activated,
            down,
            residual + down,
        )


reference = Qwen3TTSModel.from_pretrained(
    "../models/source-0.6b",
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="eager",
)
talker = reference.model.talker
ids = reference.processor.tokenizer("Hello.", return_tensors="pt")["input_ids"][:, :1]
with torch.inference_mode():
    x = (
        talker.text_projection(talker.get_text_embeddings()(ids))
        .transpose(1, 2)
        .unsqueeze(2)
        .half()
    )
    probe = Probe(DecoderLayer(talker.model.layers[0], talker.config)).eval().half()
    expected = [v.float().numpy() for v in probe(x)]
    traced = torch.jit.trace(probe, x)
names = [
    "normalized",
    "value",
    "projected",
    "residual",
    "post",
    "gate",
    "up",
    "activated",
    "down",
    "output",
]
model = ct.convert(
    traced,
    minimum_deployment_target=ct.target.macOS15,
    inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
    outputs=[ct.TensorType(name=name, dtype=np.float16) for name in names],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_ONLY,
)
model.save("../models/probe-balanced.mlpackage")
for units in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]:
    native = ct.models.MLModel(
        "../models/probe-balanced.mlpackage", compute_units=units
    )
    actual = native.predict({"x": x.numpy()})
    print(
        json.dumps(
            {
                "units": units.name,
                "errors": {
                    name: float(
                        np.linalg.norm(actual[name].astype(np.float32) - target)
                        / np.linalg.norm(target)
                    )
                    for name, target in zip(names, expected)
                },
            }
        )
    )
