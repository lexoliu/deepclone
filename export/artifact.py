"""Shared Core ML conversion, weight compression, and placement reporting."""

import json
import shutil
from collections import Counter

import coremltools as ct
import numpy as np
import torch
from coremltools.models.compute_plan import MLComputePlan
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)


def save(model, output, weight_bits):
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
    package = output.with_suffix(".mlpackage")
    model.save(str(package))
    ct.models.utils.compile_model(str(package), destination_path=str(output))
    shutil.rmtree(package)
    report = placement(output)
    report["weight_bits"] = weight_bits
    report["compiled_bytes"] = sum(
        p.stat().st_size for p in output.rglob("*") if p.is_file()
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


def placement(path):
    plan = MLComputePlan.load_from_path(
        str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
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
        "requested_compute_units": "CPU_AND_NE",
        "evidence": "compiler compute plan; not runtime utilization",
        "operations": [
            {"operator": operator, "preferred_device": device, "count": count}
            for (operator, device), count in sorted(counts.items())
        ],
    }


def convert(module, inputs, names, outputs, output, weight_bits, states=()):
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
                shape=tuple(value.shape),
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
    save(model, output, weight_bits)
