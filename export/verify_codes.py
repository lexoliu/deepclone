"""Decode identical exported acoustic codes across official and Core ML paths."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSTokenizer

from streaming_audio import StreamingAudioDecoder
from verify_audio import lowered_decode, metrics, native_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--codes", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--waveforms", type=Path, help="Optional Rust WAV outputs for comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.decoder.with_suffix(".contract.json").read_text())
    capacity = contract["capacity"]
    # Published tokenizer chunked_decode boundaries, shared with Rust inference.
    chunk_size, left_context = 300, 25
    if capacity < chunk_size + left_context:
        raise ValueError("Decoder capacity cannot hold a full chunk and context")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(args.source / "speech_tokenizer"),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    codec = tokenizer.model
    lowered = StreamingAudioDecoder(codec.decoder).eval()
    native = ct.models.CompiledMLModel(
        str(args.decoder), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    with torch.inference_mode():
        for path in args.codes:
            generation = json.loads(path.read_text())
            exported = generation["codes"]
            groups = exported["groups"]
            if groups != contract["code_groups"]:
                raise ValueError("Code groups differ from decoder contract")
            codes = torch.tensor(exported["values"], dtype=torch.int64)
            codes = codes.reshape(-1, groups).T.unsqueeze(0)
            offset = exported["reference_frames"] * codec.decode_upsample_rate
            if not 0 <= offset < codes.shape[-1] * codec.decode_upsample_rate:
                raise ValueError("Reference boundary leaves no generated audio")
            native_parts = []
            for start in range(0, codes.shape[-1], chunk_size):
                end = min(start + chunk_size, codes.shape[-1])
                context = min(start, left_context)
                chunk = codes[..., start - context : end]
                begin_sample = context * codec.decode_upsample_rate
                end_sample = chunk.shape[-1] * codec.decode_upsample_rate
                native_parts.append(
                    native_decode(native, chunk, contract).ravel()[
                        begin_sample:end_sample
                    ]
                )
            actual = np.concatenate(native_parts)[offset:]
            expected = codec.decoder.chunked_decode(codes).numpy().ravel()[offset:]
            lowered_parts = []
            for start in range(0, codes.shape[-1], chunk_size):
                end = min(start + chunk_size, codes.shape[-1])
                context = min(start, left_context)
                chunk = codes[..., start - context : end]
                lowered_parts.append(
                    lowered_decode(lowered, chunk, contract).ravel()[
                        context * codec.decode_upsample_rate :
                    ]
                )
            rewritten = np.concatenate(lowered_parts)[offset:]
            name = path.name.removesuffix(".codes.json")
            rate = contract["sample_rate"]
            reports[name] = {
                "segments": generation["segments"],
                "frames": codes.shape[-1],
                "reference_frames": exported["reference_frames"],
                "incremental_fp32": metrics(rewritten, expected),
                "coreml_fp16": metrics(actual, expected),
                "coreml_last_second": metrics(actual[-rate:], expected[-rate:]),
            }
            if args.waveforms:
                rust, rust_rate = sf.read(
                    args.waveforms / (name + ".wav"), dtype="float32"
                )
                if rust_rate != rate or rust.shape != actual.shape:
                    raise ValueError("Rust waveform shape or sample rate differs")
                reports[name]["rust_vs_coreml_max_absolute"] = float(
                    np.max(np.abs(rust - actual))
                )
            expected.astype("<f4").tofile(args.output / f"{name}-official-fp32.f32le")
            for suffix, audio in [
                ("official-fp32", expected),
                ("lowered-fp32", rewritten),
                ("coreml-fp16", actual),
            ]:
                sf.write(
                    args.output / f"{name}-{suffix}.wav", audio, rate, subtype="FLOAT"
                )
            (args.output / "metrics.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            print(name, reports[name], flush=True)


if __name__ == "__main__":
    main()
