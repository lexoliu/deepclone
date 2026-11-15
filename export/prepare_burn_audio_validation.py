"""Generate private speaker and codec fixtures from a real recording."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from safetensors.torch import save_file

from audio_models import AudioEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples, rate = sf.read(args.reference, dtype="float32")
    if rate != 24000 or samples.ndim != 1:
        raise ValueError("Reference must be mono 24 kHz")
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    codec = reference.model.speech_tokenizer.model
    with torch.inference_mode():
        audio = torch.from_numpy(samples).reshape(1, 1, -1)
        mel = mel_spectrogram(
            audio[:, 0], 1024, 128, rate, 256, 1024, 0, 12000
        ).transpose(1, 2)
        speaker = reference.model.speaker_encoder(mel)
        encoder = AudioEncoder(codec, samples.size)
        features = {}
        handle = encoder.model.downsample.register_forward_hook(
            lambda module, inputs, output: features.update(encoded=output)
        )
        codes = encoder(audio)
        handle.remove()
        tensors = {
            "audio": audio,
            "mel": mel,
            "speaker": speaker,
            "codes": codes,
            **features,
        }
        # Exercise both short causal spans and the decoder's 72-frame window.
        for frames in (8, 80):
            span = codes[:, :, :frames].contiguous()
            tensors[f"codes.{frames}"] = span
            tensors[f"waveform.{frames}"] = codec.decoder(span).contiguous()
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        args.output / "audio-fixtures.safetensors",
    )
    (args.output / "model-config.json").write_text(
        json.dumps(reference.model.config.to_dict(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
