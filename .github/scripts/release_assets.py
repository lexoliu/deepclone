"""Collect only cargo-dist's declared upload files from local and global builds."""

import json
import shutil
import sys
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
    collect(
        [json.loads(Path(path).read_text()) for path in sys.argv[1:]],
        Path("target/release-assets"),
    )
