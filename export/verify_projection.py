"""Compare an exported projection head with the current graph and official weights."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import torch
from safetensors import safe_open

from artifact import convert
from transformer import OutputHead
from verify_audio import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    config = json.loads((args.source / "config.json").read_text())["talker_config"]
    with safe_open(args.source / "model.safetensors", framework="pt") as source:
        norm = SimpleNamespace(
            weight=source.get_tensor("talker.model.norm.weight").float(),
            variance_epsilon=config["rms_norm_eps"],
        )
        head = SimpleNamespace(
            weight=source.get_tensor("talker.codec_head.weight").float(), bias=None
        )
    hidden = torch.sin(torch.arange(head.weight.shape[1]).float()).reshape(1, -1, 1, 1)
    hidden = hidden.half()
    with torch.inference_mode():
        x = hidden.flatten(1).float()
        normalized = x * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + norm.variance_epsilon
        )
        normalized = normalized * norm.weight
        expected = torch.nn.functional.linear(normalized, head.weight).numpy().ravel()
        module = OutputHead(norm, head).eval()
        torch.testing.assert_close(
            module(hidden.float())[1].flatten(),
            torch.from_numpy(expected),
            rtol=1e-4,
            atol=1e-4,
        )
        convert(
            module.half(),
            (hidden,),
            ["hidden"],
            ["normalized", "logits"],
            args.output,
            16,
        )
    models = {
        "baseline": ct.models.CompiledMLModel(
            str(args.baseline), compute_units=ct.ComputeUnit.CPU_AND_NE
        ),
        "candidate": ct.models.CompiledMLModel(
            str(args.output), compute_units=ct.ComputeUnit.CPU_AND_NE
        ),
    }
    inputs = {"hidden": hidden.numpy()}
    reports, timings = {}, {name: [] for name in models}
    for name, model in models.items():
        output = model.predict(inputs)["logits"].astype(np.float32).ravel()
        reports[name] = metrics(output, expected)
    # Alternate order to reduce systematic ordering bias in this short local probe.
    for iteration in range(args.iterations):
        order = list(models) if iteration % 2 == 0 else list(reversed(models))
        for name in order:
            start = time.perf_counter()
            models[name].predict(inputs)
            timings[name].append(time.perf_counter() - start)
    for name, values in timings.items():
        reports[name]["median_seconds"] = statistics.median(values)
        reports[name]["iterations"] = len(values)
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
