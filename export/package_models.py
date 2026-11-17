"""Package inference artifacts without recordings, generated audio, or local caches."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inputs(transformers, audio):
    manifest = json.loads((transformers / "transformers.json").read_text())
    if manifest["weight_bits"] != 16:
        raise ValueError("Published transformer weights must be FP16")
    components = [
        manifest["text_embedding"],
        manifest["codec_frame"],
        *manifest["codec_embeddings"],
    ]
    if manifest["predictor_projection"]:
        components.append(manifest["predictor_projection"])
    for decoder in manifest["decoders"].values():
        components.extend(decoder["chunks"])
        components.extend(decoder["heads"])
    selected = {
        "transformers": (
            transformers,
            ["transformers.json", "config.json", "tokenizer.json", *components],
        ),
        "audio": (audio, ["mel.json"]),
    }
    for component, precision in [
        ("encoder", "fp32"),
        ("speaker", "fp32"),
        ("decoder", "fp16"),
    ]:
        contract = f"{component}.contract.json"
        metadata = json.loads((audio / contract).read_text())
        if metadata["compute_precision"] != precision:
            raise ValueError(f"{component} must use {precision}")
        if component == "decoder" and metadata.get("execution") != "stateful_frame":
            raise ValueError(
                "Published waveform decoder must use stateful_frame execution"
            )
        selected["audio"][1].extend([contract, f"{component}.mlmodelc"])
    files = {}
    for prefix, (root, names) in selected.items():
        for name in sorted(set(names)):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid model path: {name}")
            path = root / relative
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        files[f"{prefix}/{relative}/{child.relative_to(path)}"] = child
            elif path.is_file():
                files[f"{prefix}/{relative}"] = path
            else:
                raise FileNotFoundError(path)
    return files


def package(transformers, audio, output):
    files = inputs(transformers, audio)
    records = [
        {"path": name, "size": path.stat().st_size, "sha256": digest(path)}
        for name, path in sorted(files.items())
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        process = subprocess.Popen(
            ["zstd", "-3", "-c"], stdin=subprocess.PIPE, stdout=destination
        )
        try:
            with tarfile.open(
                fileobj=process.stdin, mode="w|", dereference=True
            ) as tar:
                for name, path in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    with path.open("rb") as source:
                        tar.addfile(info, source)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("zstd compression failed")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    return {
        "file": output.name,
        "sha256": digest(output),
        "size": output.stat().st_size,
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = package(args.transformers, args.audio, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
