"""Collect only cargo-dist's declared upload files from local and global builds."""

import argparse
import json
import shutil
from pathlib import Path


def collect(manifests, output):
    sources = {}
    for manifest in manifests:
        for name in manifest["upload_files"]:
            path = Path(name)
            if path.name in sources and sources[path.name] != path:
                raise ValueError(f"Conflicting release artifact: {path.name}")
            sources[path.name] = path
    if not sources:
        raise ValueError("cargo-dist did not produce release artifacts")
    output.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        shutil.copyfile(source, output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    collect(
        [json.loads(path.read_text()) for path in args.manifests],
        args.output,
    )
