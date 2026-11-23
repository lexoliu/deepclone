"""Shared Core ML conversion, weight compression, and placement reporting."""

import json
import tempfile
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.models.compute_plan import MLComputePlan
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)


def enumerated_axis(value, axes, lengths):
    default = tuple(value.shape)
    if len(lengths) == 1:
        return default
    # Examples use a single frame or token. In a sliding attention mask,
    # the key axis incorporates each additional query as well.
    shapes = []
    for length in lengths:
        shape = list(default)
        for dimension in axes:
            shape[dimension] += length - 1
        shapes.append(shape)
    return ct.EnumeratedShapes(shapes, default=default)


def save(model, output, weight_bits, *, placement_units=ct.ComputeUnit.CPU_AND_NE):
    if output.exists():
        raise FileExistsError(output)
    if weight_bits in (4, 8):
        model = linear_quantize_weights(
            model,
            config=OptimizationConfig(
                global_config=OpLinearQuantizerConfig(
                    dtype=f"int{weight_bits}", granularity="per_channel"
                )
            ),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-", dir=output.parent
    ) as staging:
        staging = Path(staging)
        package = staging / output.with_suffix(".mlpackage").name
        compiled = staging / output.name
        model.save(str(package))
        ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        report = placement(compiled, placement_units)
        report["weight_bits"] = weight_bits
        report["compiled_bytes"] = sum(
            p.stat().st_size for p in compiled.rglob("*") if p.is_file()
        )
        report_path = staging / output.with_suffix(".json").name
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        compiled.rename(output)
        report_path.replace(output.with_suffix(".json"))


def placement(path, units):
    plan = MLComputePlan.load_from_path(str(path), compute_units=units)
    counts = Counter()

    def visit(block):
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            device = (
                type(usage.preferred_compute_device).__name__
                if usage
                else "unspecified"
            )
            counts[(operation.operator_name, device)] += 1
            for child in operation.blocks:
                visit(child)

    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    return {
        "requested_compute_units": units.name,
        "evidence": "compiler compute plan; not runtime utilization",
        "operations": [
            {"operator": operator, "preferred_device": device, "count": count}
            for (operator, device), count in sorted(counts.items())
        ],
    }


def convert(
    module,
    inputs,
    names,
    outputs,
    output,
    weight_bits,
    states=(),
    shapes=None,
    *,
    placement_units=ct.ComputeUnit.CPU_AND_NE,
):
    module.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, inputs, check_trace=False)
    if states:
        module.reset()
    model = ct.convert(
        traced,
        minimum_deployment_target=ct.target.macOS15,
        inputs=[
            ct.TensorType(
                name=name,
                shape=(shapes or {}).get(name, tuple(value.shape)),
                dtype=np.int32 if value.dtype == torch.int32 else np.float16,
            )
            for name, value in zip(names, inputs)
        ],
        outputs=[ct.TensorType(name=name, dtype=np.float16) for name in outputs],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(value.shape), dtype=np.float16),
                name=name,
            )
            for name, value in states
        ],
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    save(model, output, weight_bits, placement_units=placement_units)
