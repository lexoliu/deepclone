"""Replay real codec inputs between independent Core ML transformer predictions."""

import argparse
import json
import subprocess
from pathlib import Path


def inputs(codes, capacity, chunk_frames):
    groups = codes["groups"]
    values = codes["values"]
    if len(values) % groups:
        raise ValueError("Incomplete acoustic frame")
    frames = len(values) // groups
    start = codes["reference_frames"]
    if not 0 <= start < frames:
        raise ValueError("Reference boundary leaves no generated frames")
    while start < frames:
        end = min(start + chunk_frames, frames)
        cursor = start
        while cursor < end:
            # The published Qwen chunked decoder uses 300-frame blocks and
            # 25 preceding context frames, including the reference prefix.
            block = cursor // 300 * 300
            context = max(0, block - 25)
            stop = min(end, block + 300)
            if stop - context > capacity:
                raise ValueError("Decoder capacity cannot hold this codec block")
            yield {
                "context": context,
                "frames": stop - context,
                "values": [
                    values[frame * groups + group]
                    for group in range(groups)
                    for frame in range(context, stop)
                ],
            }
            cursor = stop
        start = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--audio-models", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--hints", choices=["default", "static"], default="default")
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("Chunk frames must be positive")
    manifest = json.loads((args.transformers / "transformers.json").read_text())
    contract = json.loads((args.audio_models / "decoder.contract.json").read_text())
    if contract["execution"] != "stateful_frame":
        raise ValueError("Interleaved replay requires stateful_frame execution")
    codes = json.loads(args.codes.read_text())["codes"]
    if codes["groups"] != contract["code_groups"]:
        raise ValueError("Acoustic code groups differ from decoder contract")
    paths = []
    for name in ["talker", "predictor"]:
        decoder = manifest["decoders"][name]
        paths.extend(decoder["chunks"])
        paths.extend(decoder["heads"])
    if manifest.get("predictor_projection"):
        paths.append(manifest["predictor_projection"])
    paths.extend(manifest["codec_embeddings"])
    paths.extend([manifest["codec_frame"], manifest["text_embedding"]])
    models = [str((args.transformers / name).resolve()) for name in paths]
    models.extend(
        str((args.audio_models / name).resolve())
        for name in ["speaker.mlmodelc", "encoder.mlmodelc"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    packed = args.output / "inputs.json"
    packed.write_text(
        json.dumps(list(inputs(codes, contract["capacity"], args.chunk_frames)))
    )
    residents = args.output / "models.json"
    residents.write_text(json.dumps(models))
    subprocess.run(
        [
            str(args.executable.resolve()),
            str((args.audio_models / "decoder.mlmodelc").resolve()),
            str(packed.resolve()),
            str((args.output / "report.json").resolve()),
            str(residents.resolve()),
            str(args.rounds),
            args.hints,
        ],
        check=True,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
