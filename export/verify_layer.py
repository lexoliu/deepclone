"""Numerical comparison against the published Qwen implementation and weights."""

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel
from transformers import DynamicCache

from transformer import DecoderLayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    talker = reference.model.talker
    text = "Hello. 你好。こんにちは。안녕하세요."
    tokens = reference.processor.tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        embeddings = talker.text_projection(talker.get_text_embeddings()(tokens))
        layer = talker.model.layers[0]
        rewritten = DecoderLayer(layer, talker.config).eval()
        config = talker.config
        context = embeddings.shape[1]
        keys = torch.zeros(1, config.num_key_value_heads, context, config.head_dim)
        values = torch.zeros(1, config.num_key_value_heads, context, config.head_dim)
        cache = DynamicCache()
        max_error = 0.0
        for index in range(context):
            hidden = embeddings[:, index : index + 1]
            position = torch.tensor([index])
            cosine, sine = talker.model.rotary_emb(
                hidden, position.reshape(1, 1, 1).expand(3, 1, 1)
            )
            expected = layer(
                hidden,
                past_key_values=cache,
                use_cache=True,
                cache_position=position,
                position_embeddings=(cosine, sine),
            )[0]
            mask = torch.full((1, 1, 1, context), torch.finfo(torch.float32).min)
            mask[..., -(index + 1) :] = 0
            actual, keys, values = rewritten(
                hidden.transpose(1, 2).unsqueeze(2),
                cosine[0].transpose(1, 2).unsqueeze(1),
                sine[0].transpose(1, 2).unsqueeze(1),
                mask,
                keys,
                values,
            )
            actual = actual.squeeze(2).transpose(1, 2)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            max_error = max(max_error, (actual - expected).abs().max().item())
    print(json.dumps({"tokens": context, "max_absolute_error": max_error}))


if __name__ == "__main__":
    main()
