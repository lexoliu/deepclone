"""Generate decoder fixtures from the pinned official model and real text."""

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    talker = reference.model.talker
    prompts = [
        "法律让人安心。",
        "Let us take a quiet walk.",
        "規則への信頼。",
        "규칙에 대한 신뢰.",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with torch.inference_mode():
        for index, text in enumerate(prompts):
            ids = reference.processor.tokenizer(text, return_tensors="pt")["input_ids"]
            hidden = talker.text_projection(talker.get_text_embeddings()(ids))
            tensors[f"input.{index}"] = hidden.contiguous()
            cache = None
            outputs = []
            for token in hidden.split(1, dim=1):
                result = talker.model(
                    inputs_embeds=token, past_key_values=cache, use_cache=True
                )
                outputs.append(result.last_hidden_state)
                cache = result.past_key_values
            tensors[f"output.{index}"] = torch.cat(outputs, dim=1).contiguous()
    save_file(tensors, args.output / "decoder-fixtures.safetensors")
    (args.output / "config.json").write_text(
        json.dumps(talker.config.to_dict(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
