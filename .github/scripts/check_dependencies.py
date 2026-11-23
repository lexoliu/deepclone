"""Reject reachable vulnerable dependencies and unexpected audit findings."""

import json
import subprocess


# Burn resolves its optional LibTorch packages into Cargo.lock even when they
# are disabled. These packages must never be part of deepclone's build graph.
EXCLUDED = {"burn-tch", "tch", "torch-sys"}
DORMANT = {
    ("zip", "0.6.6"): {
        "RUSTSEC-2026-0429",
        "RUSTSEC-2026-0367",
        "RUSTSEC-2026-0368",
        "RUSTSEC-2026-0369",
        "RUSTSEC-2026-0548",
        "RUSTSEC-2026-0545",
    },
    ("zstd-safe", "5.0.2+zstd.1.5.2"): {
        "RUSTSEC-2026-0403",
        "RUSTSEC-2026-0395",
        "RUSTSEC-2026-0402",
        "RUSTSEC-2026-0406",
        "RUSTSEC-2026-0404",
        "RUSTSEC-2026-0400",
        "RUSTSEC-2026-0399",
        "RUSTSEC-2026-0401",
        "RUSTSEC-2026-0397",
    },
}


def validate(active, audit):
    forbidden = sorted(name for name, _ in active if name in EXCLUDED)
    if forbidden:
        raise ValueError(f"LibTorch dependencies are enabled: {forbidden}")
    findings = list(audit["vulnerabilities"]["list"])
    for items in audit["warnings"].values():
        findings.extend(items)
    for finding in findings:
        package = finding["package"]
        key = (package["name"], package["version"])
        advisory = finding["advisory"]["id"]
        if key == ("paste", "1.0.15") and advisory == "RUSTSEC-2024-0436":
            continue
        if key not in active and advisory in DORMANT.get(key, set()):
            continue
        raise ValueError(f"Unresolved advisory: {key} {advisory}")


def main():
    # Cargo metadata includes optional lockfile edges; cargo tree evaluates
    # enabled features and reports the actual normal/build dependency graph.
    tree = subprocess.check_output(
        ["cargo", "tree", "--locked", "--all-features", "--target", "all",
         "--edges", "normal,build,dev", "--prefix", "none", "--format", "{p}"],
        text=True,
    )
    active = set()
    for line in tree.splitlines():
        name, version, *_ = line.split()
        if not version.startswith("v"):
            raise ValueError(f"Unexpected cargo tree entry: {line}")
        active.add((name, version[1:]))
    result = subprocess.run(["cargo", "audit", "--json"], capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode())
    validate(active, json.loads(result.stdout))


if __name__ == "__main__":
    main()
