"""Validate conditioning and codec graphs against real reference audio."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

from audio_models import AudioEncoder
from streaming_audio import StreamingAudioDecoder, metadata


def batches(frames, contract):
    sizes = contract.get("batch_frames", [1])
    offset = 0
    while offset < frames:
        count = next(size for size in reversed(sizes) if size <= frames - offset)
        yield offset, count
        offset += count


def native_decode(native, codes, contract):
    if contract["execution"] != "stateful_frame":
        raise ValueError("Waveform verification requires stateful_frame execution")
    state = native.make_state()
    parts = []
    for index, count in batches(codes.shape[-1], contract):
        values = metadata(
            contract["rotary_frequencies"],
            contract["rotary_scaling"],
            contract["attention_window"],
            index,
            torch.float16,
            count,
        )
        inputs = {"codes": codes[..., index : index + count].numpy().astype(np.int32)}
        inputs.update(
            (name, value.numpy())
            for name, value in zip(["cosine", "sine", "mask"], values)
        )
        parts.append(native.predict(inputs, state=state)["audio"])
    return np.concatenate(parts, axis=-1)


def lowered_decode(module, codes, contract):
    module.reset()
    return torch.cat(
        [
            module(
                codes[..., index : index + count],
                *metadata(
                    contract["rotary_frequencies"],
                    contract["rotary_scaling"],
                    contract["attention_window"],
                    index,
                    frames=count,
                ),
            )
            for index, count in batches(codes.shape[-1], contract)
        ],
        dim=-1,
    ).numpy()


def metrics(actual, expected):
    actual, expected = (
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
    )
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite audio model output")
    return {
        "relative_error": float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        ),
        "cosine": float(
            np.dot(actual.ravel(), expected.ravel())
            / (np.linalg.norm(actual) * np.linalg.norm(expected))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compute-units", choices=["CPU_ONLY", "CPU_AND_NE", "both"], default="both"
    )
    args = parser.parse_args()
    units_to_check = (
        [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]
        if args.compute_units == "both"
        else [getattr(ct.ComputeUnit, args.compute_units)]
    )
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    audio, rate = sf.read(args.reference, dtype="float32")
    if rate != 24000 or audio.ndim != 1:
        raise ValueError("Reference must be mono 24kHz")
    waveform = torch.from_numpy(audio)[None]
    codec = reference.model.speech_tokenizer.model
    with torch.inference_mode():
        mel = mel_spectrogram(
            waveform, 1024, 128, 24000, 256, 1024, 0, 12000
        ).transpose(1, 2)
        speaker = reference.model.speaker_encoder(mel).numpy()
        codes = codec.encoder.encode(waveform[:, None], return_dict=True).audio_codes[
            :, : codec.encoder_valid_num_quantizers
        ]
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output / "reference.npz",
            mel=mel.numpy(),
            speaker=speaker,
            codes=codes.numpy(),
        )
        reports = {}

        def record(name, value):
            reports[name] = value
            (args.output / "metrics.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            print(name, value, flush=True)

        for units in units_to_check:
            native = ct.models.CompiledMLModel(
                str(args.models / "speaker.mlmodelc"), compute_units=units
            )
            actual = native.predict({"mel": mel.numpy()})["speaker"]
            record("speaker_" + units.name, metrics(actual, speaker))
        contract = json.loads((args.models / "decoder.contract.json").read_text())
        capacity = contract["capacity"]
        short = codes[..., :capacity]
        expected = codec.decoder(short).numpy()
        rewritten = lowered_decode(
            StreamingAudioDecoder(codec.decoder), short, contract
        )
        record("decoder_equations", metrics(rewritten, expected))
        for units in units_to_check:
            native = ct.models.CompiledMLModel(
                str(args.models / "decoder.mlmodelc"), compute_units=units
            )
            actual = native_decode(native, short, contract)
            record("decoder_" + units.name, metrics(actual, expected))
            sf.write(
                args.output / ("decoded_reference_" + units.name + ".wav"),
                actual.ravel(),
                rate,
                subtype="FLOAT",
            )
        # Compare padded and original encoder outputs before adopting a fixed
        # input capacity. The final downsample has replicate boundary padding.
        capacity = (
            (audio.size + codec.encode_downsample_rate - 1)
            // codec.encode_downsample_rate
        ) * codec.encode_downsample_rate
        padded = torch.nn.functional.pad(waveform[:, None], (0, capacity - audio.size))
        encoded = AudioEncoder(codec, capacity)(padded)
        record("encoder_padded_code_match", float((encoded == codes).float().mean()))
        exact = AudioEncoder(codec, audio.size)(waveform[:, None])
        record("encoder_real_length_code_match", float((exact == codes).float().mean()))
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
