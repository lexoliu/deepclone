"""Compare a compiled stateful decoder chunk with its PyTorch equations."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from transformers import DynamicCache

from transformer import DecoderChunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--minimum-cosine", type=float)
    parser.add_argument("--maximum-relative-error", type=float)
    args = parser.parse_args()
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    talker = reference.model.talker
    tokens = reference.processor.tokenizer(
        "Hello. 你好。こんにちは。안녕하세요.", return_tensors="pt"
    )["input_ids"]
    chunk = (
        DecoderChunk(talker.model.layers[: args.layers], talker.config, args.context)
        .eval()
        .half()
    )
    samples, expected, half_expected = [], [], []
    reference_cache = DynamicCache()
    expected_states = []
    with torch.inference_mode():
        embeddings = talker.text_projection(talker.get_text_embeddings()(tokens))
        for index in range(embeddings.shape[1]):
            hidden = embeddings[:, index : index + 1]
            positions = torch.tensor([index]).reshape(1, 1, 1).expand(3, 1, 1)
            cosine, sine = talker.model.rotary_emb(hidden, positions)
            mask = torch.full(
                (1, 1, 1, args.context),
                torch.finfo(torch.float16).min,
                dtype=torch.float16,
            )
            mask[..., -(index + 1) :] = 0
            inputs = (
                hidden.transpose(1, 2).unsqueeze(2).half(),
                cosine[0].transpose(1, 2).unsqueeze(1).half(),
                sine[0].transpose(1, 2).unsqueeze(1).half(),
                mask,
            )
            half_expected.append(chunk(*inputs).float().numpy())
            expected_states.append(
                {name: buffer.numpy().copy() for name, buffer in chunk.named_buffers()}
            )
            official = inputs[0].squeeze(2).transpose(1, 2).float()
            rounded_cosine = (
                inputs[1]
                .squeeze(1)
                .transpose(1, 2)
                .float()
                .unsqueeze(0)
                .expand(3, -1, -1, -1)
            )
            rounded_sine = (
                inputs[2]
                .squeeze(1)
                .transpose(1, 2)
                .float()
                .unsqueeze(0)
                .expand(3, -1, -1, -1)
            )
            for layer in talker.model.layers[: args.layers]:
                official = layer(
                    official,
                    past_key_values=reference_cache,
                    use_cache=True,
                    cache_position=torch.tensor([index]),
                    position_embeddings=(rounded_cosine, rounded_sine),
                )[0]
            expected.append(official.transpose(1, 2).unsqueeze(2).numpy())
            samples.append(
                dict(
                    zip(
                        ["hidden", "cosine", "sine", "mask"],
                        [tensor.numpy() for tensor in inputs],
                    )
                )
            )
    half_errors = [
        float(np.linalg.norm(a - b) / np.linalg.norm(b))
        for a, b in zip(half_expected, expected)
    ]
    reports = [
        {"compute_units": "PYTORCH_FP16", "maximum_relative_error": max(half_errors)}
    ]
    failures = []
    for units in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]:
        native = ct.models.CompiledMLModel(str(args.model), compute_units=units)
        state = native.make_state()
        minimum_cosine, maximum_relative = 1.0, 0.0
        state_errors = {}
        for index, (sample, target) in enumerate(zip(samples, expected)):
            actual = native.predict(sample, state=state)["hidden_out"].astype(
                np.float32
            )
            for name, expected_state in expected_states[index].items():
                actual_state = state.read_state(name).astype(np.float32)
                expected_state = expected_state.astype(np.float32)
                error = float(
                    np.linalg.norm(actual_state - expected_state)
                    / np.linalg.norm(expected_state)
                )
                state_errors[name] = max(state_errors.get(name, 0.0), error)
            if not np.isfinite(actual).all():
                raise AssertionError("Core ML produced nonfinite decoder output")
            cosine = float(
                np.dot(actual.ravel(), target.ravel())
                / (np.linalg.norm(actual) * np.linalg.norm(target))
            )
            relative = float(np.linalg.norm(actual - target) / np.linalg.norm(target))
            minimum_cosine = min(minimum_cosine, cosine)
            maximum_relative = max(maximum_relative, relative)
            if (args.minimum_cosine is not None and cosine < args.minimum_cosine) or (
                args.maximum_relative_error is not None
                and relative > args.maximum_relative_error
            ):
                failures.append(
                    {
                        "units": units.name,
                        "token": index,
                        "cosine": cosine,
                        "relative": relative,
                        "target_norm": float(np.linalg.norm(target)),
                        "actual_norm": float(np.linalg.norm(actual)),
                    }
                )
        reports.append(
            {
                "compute_units": units.name,
                "tokens": len(samples),
                "minimum_cosine": minimum_cosine,
                "maximum_relative_error": maximum_relative,
                "state_errors": state_errors,
            }
        )
    print(json.dumps({"reports": reports, "failures": failures}))
    if failures:
        raise AssertionError("Core ML chunk exceeds numerical tolerance")


if __name__ == "__main__":
    main()
