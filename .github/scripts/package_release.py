"""Build and validate the same cargo-dist artifacts for previews and releases."""

import argparse
import json
import subprocess
from pathlib import Path

from check_release_artifacts import check
from release_assets import collect


def package(output, tag):
    def build(kind):
        command = ["dist", "build", f"--artifacts={kind}", "--output-format=json"]
        if tag is not None:
            command.extend(["--tag", tag])
        return json.loads(subprocess.check_output(command, text=True))

    local = build("local")
    # cargo-dist merges manifests in its distribution directory into the global
    # build, supplying archive checksums to the installer and aggregate list.
    manifest = Path("target/distrib/aarch64-apple-darwin-dist-manifest.json")
    manifest.write_text(json.dumps(local))
    global_artifacts = build("global")
    collect([local, global_artifacts], output)
    check(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag")
    args = parser.parse_args()
    package(args.output, args.tag)
