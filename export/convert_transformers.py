"""Export both Qwen autoregressive networks and their embedding/head contracts."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel
from safetensors.torch import save_file

from artifact import convert, enumerated_axis
from provenance import snapshot, verify_or_record
from transformer import (
    CodecEmbedding,
    CodecFrame,
    DecoderChunk,
    OutputHead,
    Projection,
    TextEmbedding,
)


def export(
    reference,
    output,
    context,
    chunk_layers,
    predictor_chunk_layers,
    weight_bits,
    sequence_lengths,
    resume=False,
    host_tables=False,
):
    talker = reference.model.talker
    manifest = {"weight_bits": weight_bits, "decoders": {}}
    token = torch.zeros((1, 1), dtype=torch.int32)

    def component(
        module,
        inputs,
        names,
        outputs,
        name,
        states=(),
        shapes=None,
        bits=weight_bits,
    ):
        path = output / (name + ".mlmodelc")
        if resume and path.exists():
            report_path = path.with_suffix(".json")
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report["weight_bits"] != bits:
                    raise ValueError(f"{name}: existing weight precision differs")
                return path.name
            # The component was interrupted before its completion report was
            # published. The run's provenance has already been checked.
            shutil.rmtree(path)
        convert(module.half(), inputs, names, outputs, path, bits, states, shapes)
        return path.name

    # TextEmbedding is a 151,936-row gather table that executes on CPU;
    # quantizing it gives no ANE speedup and stalls Core ML gather constant folding.
    manifest["text_embedding"] = component(
        TextEmbedding(talker),
        (token,),
        ["token"],
        ["hidden"],
        "text-embedding",
        bits=16,
    )
    if host_tables:
        tables = {}
        for index, embedding in enumerate(
            [
                talker.get_input_embeddings(),
                *talker.code_predictor.get_input_embeddings(),
            ]
        ):
            tables[f"codec_embeddings.{index}"] = (
                embedding.weight.detach().to(torch.float16).contiguous()
            )
        tables["talker_norm"] = (
            talker.model.norm.weight.detach().to(torch.float16).contiguous()
        )
        tables["talker_head"] = (
            talker.codec_head.weight.detach().to(torch.float16).contiguous()
        )
        tables["predictor_norm"] = (
            talker.code_predictor.model.norm.weight.detach()
            .to(torch.float16)
            .contiguous()
        )
        for index, head in enumerate(talker.code_predictor.lm_head):
            tables[f"predictor_heads.{index}"] = (
                head.weight.detach().to(torch.float16).contiguous()
            )
        save_file(tables, str(output / "tables.safetensors"))
        manifest["tables"] = "tables.safetensors"
        manifest["codec_embeddings"] = []
        manifest["codec_frame"] = None
    else:
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
    for name, decoder, capacity, heads, lengths in [
        ("talker", talker, context, [talker.codec_head], sequence_lengths),
        (
            "predictor",
            talker.code_predictor,
            # Two prefix positions produce the first residual code; only the
            # remaining groups feed another token. Minimum positions = groups.
            talker.config.num_code_groups,
            talker.code_predictor.lm_head,
            [1],
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
        shapes = {
            feature: enumerated_axis(value, (axis,), lengths)
            for feature, value, axis in [
                ("hidden", hidden, 3),
                ("cosine", cosine, 3),
                ("sine", sine, 3),
                ("mask", mask, 2),
            ]
        }
        layers_per_chunk = chunk_layers if name == "talker" else predictor_chunk_layers
        for start in range(0, len(decoder.model.layers), layers_per_chunk):
            chunk = DecoderChunk(
                decoder.model.layers[start : start + layers_per_chunk], config, capacity
            )
            chunks.append(
                component(
                    chunk,
                    (hidden, cosine, sine, mask),
                    ["hidden", "cosine", "sine", "mask"],
                    ["hidden_out"],
                    f"{name}-{start}",
                    list(chunk.named_buffers()),
                    shapes,
                )
            )
        manifest["decoders"][name] = {
            "chunks": chunks,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "context": capacity,
            "sequence_lengths": lengths,
            "heads": []
            if host_tables
            else [
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
    parser.add_argument(
        "--predictor-chunk-layers",
        type=int,
        help="Defaults to the Talker layer group size",
    )
    parser.add_argument("--weight-bits", type=int, choices=[4, 8, 16], default=16)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[1],
        help="Ordered unique Talker token lengths, beginning with 1, for batched prefill",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed components from the same source and settings",
    )
    parser.add_argument(
        "--host-tables",
        action="store_true",
        help="Store codec embeddings and output heads in tables.safetensors for native host execution",
    )
    args = parser.parse_args()
    if args.context <= 0 or args.chunk_layers <= 0:
        parser.error("context and chunk-layers must be positive")
    if args.predictor_chunk_layers is not None and args.predictor_chunk_layers <= 0:
        parser.error("predictor-chunk-layers must be positive")
    predictor_chunk_layers = args.predictor_chunk_layers or args.chunk_layers
    if (
        args.sequence_lengths != sorted(set(args.sequence_lengths))
        or args.sequence_lengths[0] != 1
        or args.sequence_lengths[-1] > args.context
    ):
        parser.error("sequence-lengths must increase from 1 and fit the context")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    verify_or_record(
        args.output,
        snapshot(
            args.source,
            {
                "context": args.context,
                "chunk_layers": args.chunk_layers,
                "predictor_chunk_layers": predictor_chunk_layers,
                "weight_bits": args.weight_bits,
                "sequence_lengths": args.sequence_lengths,
                "host_tables": args.host_tables,
            },
        ),
        args.resume,
    )
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
        predictor_chunk_layers,
        args.weight_bits,
        args.sequence_lengths,
        args.resume,
        args.host_tables,
    )


if __name__ == "__main__":
    main()
