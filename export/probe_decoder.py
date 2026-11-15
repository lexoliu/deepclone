"""Locate waveform error with identical real codes and decoder-stage outputs."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from coremltools.converters.mil.mil.scope import ScopeSource
from qwen_tts import Qwen3TTSModel
from torch import nn

from artifact import save
from audio_models import causal_mask, lower_decoder


class Stages(nn.Module):
    def __init__(self, decoder, frames):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("full", causal_mask(frames))
        self.register_buffer(
            "sliding", causal_mask(frames, decoder.config.sliding_window)
        )
        self.names = ["quantizer", "pre_conv", "transformer"]
        self.names += [f"upsample_{i}" for i in range(len(decoder.upsample))]
        for i, block in enumerate(decoder.decoder):
            if i == 1:
                self.names += [
                    f"wave_{i}_part_{j}" for j in range(len(block.block) - 1)
                ]
            self.names.append(f"wave_{i}")

    def forward(self, codes):
        hidden = self.decoder.quantizer.decode(codes)
        outputs = [hidden]
        hidden = self.decoder.pre_conv(hidden)
        outputs.append(hidden)
        hidden = self.decoder.pre_transformer(
            inputs_embeds=hidden.transpose(1, 2),
            attention_mask={
                "full_attention": self.full,
                "sliding_attention": self.sliding,
            },
            use_cache=False,
        ).last_hidden_state.transpose(1, 2)
        outputs.append(hidden)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
            outputs.append(hidden)
        for i, block in enumerate(self.decoder.decoder):
            if i == 1:
                for j, part in enumerate(block.block):
                    hidden = part(hidden)
                    if j + 1 < len(block.block):
                        outputs.append(hidden)
            else:
                hidden = block(hidden)
            outputs.append(hidden)
        return tuple(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--keep-fp32", nargs="*", default=[])
    parser.add_argument("--polyphase", action="store_true")
    parser.add_argument(
        "--mode", choices=["convert", "reference", "evaluate"], default="convert"
    )
    parser.add_argument(
        "--compute-units", choices=["CPU_ONLY", "CPU_AND_NE"], default="CPU_AND_NE"
    )
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.output, args.compute_units)
        return
    args.output.mkdir(parents=True, exist_ok=args.mode == "reference")
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    module = Stages(reference.model.speech_tokenizer.model.decoder, args.frames).eval()
    codes = torch.from_numpy(
        np.load(args.codes)["codes"][..., : args.frames].astype(np.int32)
    )
    with torch.inference_mode():
        expected = [x.numpy() for x in module(codes)]
        np.savez(
            args.output / "reference.npz",
            codes=codes.numpy(),
            **dict(zip(module.names, expected)),
        )
        if args.mode == "reference":
            return
        if args.polyphase:
            lower_decoder(module.decoder)
            for actual, target in zip(module(codes), expected):
                torch.testing.assert_close(
                    actual, torch.from_numpy(target), rtol=1e-3, atol=1e-4
                )
        traced = torch.jit.trace(module, (codes,), check_trace=False)
    kept = set()

    def precision(op):
        types = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
        matches = set(types).intersection(args.keep_fp32)
        kept.update(matches)
        return not matches

    model = ct.convert(
        traced,
        minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name="codes", shape=tuple(codes.shape), dtype=np.int32)],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in module.names],
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=precision),
        skip_model_load=True,
    )
    if kept != set(args.keep_fp32):
        raise ValueError(
            f"Requested precision scopes not found: {set(args.keep_fp32) - kept}"
        )
    path = args.output / "stages.mlmodelc"
    save(model, path, 16)
    sf.write(
        args.output / "pytorch.wav",
        expected[-1].ravel().clip(-1, 1),
        24000,
        subtype="FLOAT",
    )


def evaluate(output, compute_units):
    reference = np.load(output / "reference.npz")
    names = [name for name in reference.files if name != "codes"]
    path = output / "stages.mlmodelc"
    reports = {}
    for units in [getattr(ct.ComputeUnit, compute_units)]:
        native = ct.models.CompiledMLModel(str(path), compute_units=units)
        outputs = native.predict({"codes": reference["codes"]})
        report = []
        for name in names:
            target = reference[name]
            actual = outputs[name]
            if not np.isfinite(actual).all():
                raise ValueError(f"Nonfinite {units.name} output at {name}")
            report.append(
                {
                    "stage": name,
                    "relative_error": float(
                        np.linalg.norm(actual - target) / np.linalg.norm(target)
                    ),
                    "maximum_absolute_error": float(np.max(np.abs(actual - target))),
                }
            )
        reports[units.name] = report
        (output / (units.name + ".json")).write_text(
            json.dumps(reports, indent=2) + "\n"
        )
        sf.write(
            output / (units.name + ".wav"),
            outputs[names[-1]].ravel().clip(-1, 1),
            24000,
            subtype="FLOAT",
        )
        print(units.name, report, flush=True)


if __name__ == "__main__":
    main()
