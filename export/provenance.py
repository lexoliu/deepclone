"""Bind resumable exports to their model inputs, implementation and settings."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(source, settings):
    exporter = Path(__file__).parent
    return {
        "source": {
            path.relative_to(source).as_posix(): digest(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
            and path.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}
            and not any(part.startswith(".") for part in path.relative_to(source).parts)
        },
        "exporter": {
            name: digest(exporter / name)
            for name in [
                "convert_transformers.py",
                "transformer.py",
                "artifact.py",
                "provenance.py",
                "pyproject.toml",
                "uv.lock",
            ]
        },
        "runtime": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in [
                    "torch",
                    "coremltools",
                    "qwen-tts",
                    "transformers",
                    "numpy",
                ]
            },
        },
        "settings": settings,
    }


def verify_or_record(output, inputs, resume):
    path = output / "export-inputs.json"
    if resume:
        if not path.exists():
            raise ValueError(
                "Cannot resume without export-inputs.json; use a new output directory"
            )
        previous = json.loads(path.read_text())
        if previous != inputs:
            changed = sorted(
                key
                for key in previous.keys() | inputs.keys()
                if previous.get(key) != inputs.get(key)
            )
            raise ValueError(
                f"Cannot resume with changed export inputs: {', '.join(changed)}"
            )
    else:
        with path.open("x") as stream:
            json.dump(inputs, stream, indent=2, sort_keys=True)
            stream.write("\n")
