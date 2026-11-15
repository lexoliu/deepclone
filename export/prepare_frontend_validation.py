"""Prepare local, private frontend parity data from a real recording."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio, rate = sf.read(args.reference, dtype="float32")
    if rate != 24000 or audio.ndim != 1:
        raise ValueError("Reference must be mono 24kHz")
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    texts = [item["text"] for item in json.loads(args.prompts.read_text())]
    texts.append("<|im_start|>assistant\n")
    expected = mel_spectrogram(
        torch.from_numpy(audio)[None], 1024, 128, rate, 256, 1024, 0, 12000
    ).transpose(1, 2)
    args.output.write_text(
        json.dumps(
            {
                "audio": audio.tolist(),
                "mel": expected.flatten().tolist(),
                "texts": [
                    {
                        "text": text,
                        "ids": tokenizer.encode(text, add_special_tokens=False),
                    }
                    for text in texts
                ],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
