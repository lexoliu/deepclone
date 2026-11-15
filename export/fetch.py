"""Fetch immutable official model inputs and verify their complete contents."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def fetch(root, item):
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size == item["size"] and digest == item["sha256"]:
            return
        raise ValueError(f"Existing model file failed verification: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    try:
        with (
            urllib.request.urlopen(item["url"], timeout=60) as source,
            partial.open("wb") as target,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
                digest.update(block)
                count += len(block)
        if count != item["size"] or digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Download failed verification: {path}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=["0.6b", "1.7b"])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name("sources.json").read_text())[args.size]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda item: fetch(args.destination, item), pins["files"]))


if __name__ == "__main__":
    main()
