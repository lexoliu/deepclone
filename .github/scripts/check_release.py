"""Check the distribution plan and the release policy without publishing."""

import json
import sys
import tomllib
from pathlib import Path


def check(plan):
    dist = tomllib.loads(Path("dist-workspace.toml").read_text())["dist"]
    policy = tomllib.loads(Path("release-plz.toml").read_text())["workspace"]
    assert dist["targets"] == ["aarch64-apple-darwin"]
    assert dist["features"] == ["cli"] and not dist["default-features"]
    assert policy["publish"] is False and policy["release_always"] is False
    assert policy["git_only"] is True and policy["git_release_draft"] is True
    assert plan["dist_version"] == dist["cargo-dist-version"]
    assert {release["app_name"] for release in plan["releases"]} == {"deepclone"}
    assert set(plan["artifacts"]) == {
        "deepclone-aarch64-apple-darwin.tar.xz",
        "deepclone-aarch64-apple-darwin.tar.xz.sha256",
        "deepclone-installer.sh",
        "sha256.sum",
        "source.tar.gz",
        "source.tar.gz.sha256",
    }


if __name__ == "__main__":
    check(json.loads(Path(sys.argv[1]).read_text()))
