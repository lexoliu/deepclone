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
            predictor = talker.code_predictor
            past = outputs[-1]
            code = talker.codec_head(past).argmax(-1)
            inputs = [predictor.small_to_mtp_projection(past)]
            inputs.append(
                predictor.small_to_mtp_projection(talker.get_input_embeddings()(code))
            )
            predicted = []
            cache = None
            for position in range(talker.config.num_code_groups + 1):
                result = predictor.model(
                    inputs_embeds=inputs[position],
                    past_key_values=cache,
                    use_cache=True,
                )
                predicted.append(result.last_hidden_state)
                cache = result.past_key_values
                if 1 <= position < talker.config.num_code_groups:
                    code = predictor.lm_head[position - 1](
                        result.last_hidden_state
                    ).argmax(-1)
                    inputs.append(
                        predictor.small_to_mtp_projection(
                            predictor.get_input_embeddings()[position - 1](code)
                        )
                    )
            tensors[f"predictor_input.{index}"] = torch.cat(inputs, dim=1).contiguous()
            tensors[f"predictor_output.{index}"] = torch.cat(
                predicted, dim=1
            ).contiguous()
    save_file(tensors, args.output / "decoder-fixtures.safetensors")
    (args.output / "predictor-config.json").write_text(
        json.dumps(talker.config.code_predictor_config.to_dict(), indent=2) + "\n"
    )
    (args.output / "config.json").write_text(
        json.dumps(talker.config.to_dict(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
