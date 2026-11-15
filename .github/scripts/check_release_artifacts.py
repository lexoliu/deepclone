"""Verify archives, the combined checksum list, and installer checksum pins."""

import hashlib
import sys
from pathlib import Path


def check(directory):
    combined = (directory / "sha256.sum").read_text()
    installer = (directory / "deepclone-installer.sh").read_text()
    checksums = list(directory.glob("*.sha256"))
    if not checksums:
        raise ValueError("No archive checksums were produced")
    for checksum in checksums:
        archive = checksum.with_suffix("")
        expected = checksum.read_text().split()[0]
        with archive.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Archive checksum mismatch: {archive.name}")
        if expected not in combined:
            raise ValueError(f"Combined checksum list omits {archive.name}")
        if archive.name.endswith(".tar.xz") and expected not in installer:
            raise ValueError(f"Installer omits checksum for {archive.name}")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
