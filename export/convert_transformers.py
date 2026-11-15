"""Export both Qwen autoregressive networks and their embedding/head contracts."""

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

from artifact import convert
from transformer import (
    CodecEmbedding,
    CodecFrame,
    DecoderChunk,
    OutputHead,
    Projection,
    TextEmbedding,
)


def export(reference, output, context, chunk_layers, weight_bits, resume=False):
    talker = reference.model.talker
    manifest = {"weight_bits": weight_bits, "decoders": {}}
    token = torch.zeros((1, 1), dtype=torch.int32)

    def component(module, inputs, names, outputs, name, states=()):
        path = output / (name + ".mlmodelc")
        if resume and path.exists():
            report = json.loads(path.with_suffix(".json").read_text())
            if report["weight_bits"] != weight_bits:
                raise ValueError(f"{name}: existing weight precision differs")
            return path.name
        convert(module.half(), inputs, names, outputs, path, weight_bits, states)
        return path.name

    manifest["text_embedding"] = component(
        TextEmbedding(talker), (token,), ["token"], ["hidden"], "text-embedding"
    )
    manifest["codec_embeddings"] = [
        component(
            CodecEmbedding(embedding),
            (token,),
            ["token"],
            ["hidden"],
            f"codec-embedding-{index}",
        )
        for index, embedding in enumerate(
            [
                talker.get_input_embeddings(),
                *talker.code_predictor.get_input_embeddings(),
            ]
        )
    ]
    manifest["codec_frame"] = component(
        CodecFrame(talker),
        (
            torch.zeros((1, talker.config.num_code_groups), dtype=torch.int32),
            torch.zeros((1, talker.config.hidden_size, 1, 1), dtype=torch.float16),
        ),
        ["codes", "text"],
        ["hidden"],
        "codec-frame",
    )
    projection = talker.code_predictor.small_to_mtp_projection
    manifest["predictor_projection"] = (
        component(
            Projection(projection),
            (torch.zeros((1, talker.config.hidden_size, 1, 1), dtype=torch.float16),),
            ["hidden"],
            ["hidden_out"],
            "predictor-projection",
        )
        if isinstance(projection, torch.nn.Linear)
        else None
    )
    for name, decoder, capacity, heads in [
        ("talker", talker, context, [talker.codec_head]),
        (
            "predictor",
            talker.code_predictor,
            talker.config.num_code_groups + 1,
            talker.code_predictor.lm_head,
        ),
    ]:
        config = decoder.config
        hidden = torch.zeros((1, config.hidden_size, 1, 1), dtype=torch.float16)
        cosine = torch.ones((1, 1, config.head_dim, 1), dtype=torch.float16)
        sine = torch.zeros_like(cosine)
        mask = torch.full(
            (1, 1, 1, capacity), torch.finfo(torch.float16).min, dtype=torch.float16
        )
        mask[..., -1] = 0
        chunks = []
        for start in range(0, len(decoder.model.layers), chunk_layers):
            chunk = DecoderChunk(
                decoder.model.layers[start : start + chunk_layers], config, capacity
            )
            chunks.append(
                component(
                    chunk,
                    (hidden, cosine, sine, mask),
                    ["hidden", "cosine", "sine", "mask"],
                    ["hidden_out"],
                    f"{name}-{start}",
                    list(chunk.named_buffers()),
                )
            )
        manifest["decoders"][name] = {
            "chunks": chunks,
            "rope_theta": config.rope_theta,
            "context": capacity,
            "heads": [
                component(
                    OutputHead(decoder.model.norm, head),
                    (hidden,),
                    ["hidden"],
                    ["normalized", "logits"],
                    f"{name}-head-{index}",
                )
                for index, head in enumerate(heads)
            ],
        }
    reference.processor.tokenizer.backend_tokenizer.save(str(output / "tokenizer.json"))
    (output / "config.json").write_text(
        reference.model.config.to_json_string(use_diff=False)
    )
    (output / "transformers.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--chunk-layers", type=int, default=4)
    parser.add_argument("--weight-bits", type=int, choices=[4, 8, 16], default=16)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed components from the same source and settings",
    )
    args = parser.parse_args()
    if args.context <= 0 or args.chunk_layers <= 0:
        parser.error("context and chunk-layers must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    reference = Qwen3TTSModel.from_pretrained(
        str(args.source),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    export(
        reference,
        args.output,
        args.context,
        args.chunk_layers,
        args.weight_bits,
        args.resume,
    )


if __name__ == "__main__":
    main()
