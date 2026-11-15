"""Produce official-model comparison audio for a locally supplied voice."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    model = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    transcript = json.loads(args.transcript.read_text())["text"]
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(args.reference), ref_text=transcript
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for item in json.loads(args.prompts.read_text()):
        audio, rate = model.generate_voice_clone(
            text=item["text"],
            language=item["language"],
            voice_clone_prompt=prompt,
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=args.max_tokens,
        )
        path = args.output / (item["language"] + ".wav")
        sf.write(path, audio[0], rate, subtype="FLOAT")
        results.append(
            {
                **item,
                "file": path.name,
                "sample_rate": rate,
                "samples": len(audio[0]),
                "backend": "official PyTorch FP32",
            }
        )
        (args.output / "samples.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
