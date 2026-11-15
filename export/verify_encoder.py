"""Compare original, lowered, and Core ML codes at the real input length."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from audio_models import AudioEncoder


def compare(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Code shapes differ: {actual.shape}, {expected.shape}")
    return {
        "code_match": float(np.mean(actual == expected)),
        "per_group_match": np.mean(actual == expected, axis=(0, 2)).tolist(),
        "mismatched_frames": np.nonzero(np.any(actual != expected, axis=(0, 1)))[
            0
        ].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    audio, rate = sf.read(args.reference, dtype="float32")
    codec = reference.model.speech_tokenizer.model
    if rate != codec.input_sample_rate or audio.ndim != 1:
        raise ValueError("Reference must be mono at the model sample rate")
    waveform = torch.from_numpy(audio)[None, None]
    reports = {}

    def record(name, actual, expected):
        reports[name] = compare(actual, expected)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        print(name, reports[name], flush=True)

    with torch.inference_mode():
        expected = (
            codec.encoder.encode(waveform, return_dict=True)
            .audio_codes[:, : codec.encoder_valid_num_quantizers]
            .numpy()
        )
        actual = AudioEncoder(codec, audio.size)(waveform).numpy()
        record("lowered_fp32", actual, expected)
        for units in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]:
            model = ct.models.CompiledMLModel(str(args.model), compute_units=units)
            actual = model.predict({"audio": waveform.numpy()})["codes"]
            record(units.name, actual, expected)


if __name__ == "__main__":
    main()
