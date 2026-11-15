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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--greedy", action="store_true")
    # Qwen3TTSModel._merge_generate_kwargs defaults, matching the Rust CLI.
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
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
            do_sample=not args.greedy,
            subtalker_dosample=not args.greedy,
            temperature=args.temperature,
            subtalker_temperature=args.temperature,
            top_k=args.top_k,
            subtalker_top_k=args.top_k,
            top_p=args.top_p,
            subtalker_top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
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
                "seed": args.seed,
                "greedy": args.greedy,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
            }
        )
        (args.output / "samples.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
