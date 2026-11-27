"""Prepare the verified browser bundle for Cloudflare Pages and R2 hosting."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

PAGES_MAX_FILE_BYTES = 25 * 1024 * 1024


def package_pages(dist: Path, wasm_out: Path, assets_origin: str) -> dict[str, str]:
    wasm_source = dist / "wasm" / "deepclone_bg.wasm"
    if not wasm_source.is_file() or wasm_source.stat().st_size == 0:
        raise ValueError(f"Missing browser WebAssembly binary: {wasm_source}")

    headers_file = dist / "_headers"
    if not headers_file.is_file():
        raise ValueError(f"Missing Cloudflare Pages _headers file: {headers_file}")
    headers_text = headers_file.read_text(encoding="utf-8")
    for required in ("Content-Security-Policy:", "Permissions-Policy:"):
        if required not in headers_text:
            raise ValueError(f"_headers omits {required}")

    digest = hashlib.sha256(wasm_source.read_bytes()).hexdigest()
    short_hash = digest[:16]
    filename = f"deepclone-{short_hash}.wasm"
    object_key = f"wasm/{filename}"
    public_url = f"{assets_origin.rstrip('/')}/{object_key}"

    wasm_out.mkdir(parents=True, exist_ok=True)
    shutil.move(str(wasm_source), str(wasm_out / filename))

    for unused_types in ("deepclone.d.ts", "deepclone_bg.wasm.d.ts"):
        candidate = dist / "wasm" / unused_types
        if candidate.exists():
            candidate.unlink()

    (dist / "wasm.json").write_text(
        json.dumps({"url": public_url}, indent=2) + "\n", encoding="utf-8"
    )

    for path in sorted(dist.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".wasm":
            raise ValueError(f"Unextracted .wasm asset remains in Pages bundle: {path}")
        size = path.stat().st_size
        if size > PAGES_MAX_FILE_BYTES:
            raise ValueError(
                f"Pages asset exceeds 25 MiB limit ({size} bytes): {path}"
            )

    metadata = {
        "file": filename,
        "key": object_key,
        "sha256": digest,
        "url": public_url,
    }
    (wasm_out / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def stale_wasm_keys(
    objects: list[dict[str, str]], active_key: str, keep: int = 2
) -> list[str]:
    ordered = sorted(
        objects,
        key=lambda item: item.get("last_modified") or item.get("uploaded") or "",
        reverse=True,
    )
    retained = {active_key}
    for item in ordered:
        if len(retained) >= keep:
            break
        key = item.get("key", "")
        if key.startswith("wasm/deepclone-") and key.endswith(".wasm"):
            retained.add(key)
    stale: list[str] = []
    for item in ordered:
        key = item.get("key", "")
        if (
            key.startswith("wasm/deepclone-")
            and key.endswith(".wasm")
            and key not in retained
        ):
            stale.append(key)
    return stale


def prune_r2_wasm(bucket: str, active_key: str, keep: int = 2) -> None:
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    query = urllib.parse.urlencode({"prefix": "wasm/deepclone-"})
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/r2/buckets/{bucket}/objects?{query}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read())
    objects = payload.get("result") or []
    for stale_key in stale_wasm_keys(objects, active_key, keep=keep):
        subprocess.run(
            [
                "bunx",
                "wrangler@4.143.1",
                "r2",
                "object",
                "delete",
                f"{bucket}/{stale_key}",
                "--remote",
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dist", type=Path, required=True)
    prepare.add_argument("--wasm-out", type=Path, required=True)
    prepare.add_argument("--assets-origin", required=True)

    prune = subparsers.add_parser("prune")
    prune.add_argument("--bucket", required=True)
    prune.add_argument("--active-key", required=True)
    prune.add_argument("--keep", type=int, default=2)

    args = parser.parse_args()
    if args.command == "prepare":
        package_pages(args.dist, args.wasm_out, args.assets_origin)
    else:
        prune_r2_wasm(args.bucket, args.active_key, args.keep)
