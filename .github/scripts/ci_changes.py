"""Map changed repository paths to the CI jobs that read them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

JOBS = (
    "dependencies",
    "workflows",
    "rust_lint",
    "rust_macos",
    "burn_linux",
    "export",
    "release_preview",
    "browser",
)

ALL_JOB_TRIGGERS = {
    ".github/workflows/ci.yml",
    ".github/scripts/ci_changes.py",
    ".github/scripts/test_ci_changes.py",
}

ZERO_SHA = "0" * 40


def classify(paths: list[str] | None) -> dict[str, bool]:
    """Return which CI jobs must run for the changed paths (`None` means run all)."""
    if paths is None:
        return {job: True for job in JOBS}

    selected = {job: False for job in JOBS}
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        if path in ALL_JOB_TRIGGERS:
            return {job: True for job in JOBS}

        if path in {"Cargo.toml", "Cargo.lock"} or path.startswith(
            ("rust-toolchain", ".cargo/")
        ):
            for job in (
                "dependencies",
                "workflows",
                "rust_lint",
                "rust_macos",
                "burn_linux",
                "release_preview",
                "browser",
            ):
                selected[job] = True
            continue

        if path.startswith("src/") or path == "build.rs":
            for job in (
                "rust_lint",
                "rust_macos",
                "burn_linux",
                "release_preview",
                "browser",
            ):
                selected[job] = True
            continue

        if path.startswith("tests/"):
            for job in ("rust_lint", "rust_macos", "burn_linux"):
                selected[job] = True
            continue

        if path.startswith("web/"):
            selected["browser"] = True
            continue

        if path.startswith("export/"):
            selected["export"] = True
            continue

        if path in {"dist-workspace.toml", "release-plz.toml"}:
            selected["workflows"] = True
            selected["release_preview"] = True
            continue

        if path.startswith(".github/"):
            selected["workflows"] = True
            if path in {
                ".github/scripts/check_dependencies.py",
                ".github/scripts/test_dependencies.py",
            }:
                selected["dependencies"] = True
            if path in {
                ".github/scripts/audit_web.ts",
                ".github/scripts/audit_web.test.ts",
            }:
                selected["browser"] = True
            if path in {
                ".github/scripts/package_release.py",
                ".github/scripts/release_assets.py",
                ".github/scripts/check_release_artifacts.py",
            }:
                selected["release_preview"] = True

    return selected


def changed_paths(event_name: str, base_sha: str, head_ref: str = "HEAD") -> list[str] | None:
    """Resolve changed paths from git, returning `None` when all jobs should run."""
    if event_name == "workflow_dispatch":
        return None
    base = base_sha.strip()
    if not base or base == ZERO_SHA:
        return None
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{base}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        return None
    diff_range = f"{base}..{head_ref}" if event_name == "push" else f"{base}...{head_ref}"
    result = subprocess.run(
        ["git", "diff", "--name-only", diff_range],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    event_name = os.environ.get("EVENT_NAME", "")
    base_sha = os.environ.get("BASE_SHA", "")
    paths = changed_paths(event_name, base_sha)
    selected = classify(paths)
    output_path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{job}={'true' if enabled else 'false'}\n" for job, enabled in selected.items()]
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
    else:
        sys.stdout.writelines(lines)


if __name__ == "__main__":
    main()
