"""Cloudflare Pages bundle preparation must enforce size and security gates."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from package_pages import PAGES_MAX_FILE_BYTES, package_pages, stale_wasm_keys


class PagesPackaging(unittest.TestCase):
    def _seed_dist(self, root: Path) -> Path:
        dist = root / "dist"
        (dist / "wasm").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
        (dist / "_headers").write_text(
            "/*\n"
            "  Content-Security-Policy: default-src 'self'\n"
            "  Permissions-Policy: microphone=(self)\n\n"
            "/wasm/*\n"
            "  Cache-Control: public, max-age=31536000, immutable\n\n"
            "/wasm.json\n"
            "  Cache-Control: public, max-age=0, must-revalidate\n",
            encoding="utf-8",
        )
        (dist / "wasm" / "deepclone.js").write_text("export default {}", encoding="utf-8")
        (dist / "wasm" / "deepclone.d.ts").write_text("export {}", encoding="utf-8")
        (dist / "wasm" / "deepclone_bg.wasm.d.ts").write_text("export {}", encoding="utf-8")
        (dist / "wasm" / "deepclone_bg.wasm").write_bytes(b"\0asm\x01\x00\x00\x00payload")
        (dist / "wasm.json").write_text(
            '{"js":"/wasm/deepclone.js","url":"/wasm/deepclone_bg.wasm"}',
            encoding="utf-8",
        )
        return dist

    def test_extracts_hashed_wasm_and_rewrites_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = self._seed_dist(root)
            expected_sha = hashlib.sha256(
                (dist / "wasm" / "deepclone_bg.wasm").read_bytes()
            ).hexdigest()
            expected_js_sha = hashlib.sha256(
                (dist / "wasm" / "deepclone.js").read_bytes()
            ).hexdigest()
            wasm_out = root / "wasm-out"

            meta = package_pages(
                dist, wasm_out, "https://assets.deepclone.lexo.cool/"
            )

            expected_name = f"deepclone-{expected_sha[:16]}.wasm"
            expected_js_name = f"deepclone-{expected_js_sha[:16]}.js"
            self.assertEqual(meta["file"], expected_name)
            self.assertEqual(meta["key"], f"wasm/{expected_name}")
            self.assertEqual(
                meta["url"],
                f"https://assets.deepclone.lexo.cool/wasm/{expected_name}",
            )
            self.assertEqual(meta["js"], f"/wasm/{expected_js_name}")
            self.assertEqual(meta["js_sha256"], expected_js_sha)
            self.assertTrue((wasm_out / expected_name).is_file())
            self.assertTrue((dist / "wasm" / expected_js_name).is_file())
            self.assertFalse((dist / "wasm" / "deepclone.js").exists())
            self.assertFalse((dist / "wasm" / "deepclone_bg.wasm").exists())
            self.assertFalse((dist / "wasm" / "deepclone.d.ts").exists())
            self.assertEqual(
                json.loads((dist / "wasm.json").read_text(encoding="utf-8")),
                {"js": meta["js"], "url": meta["url"]},
            )

    def test_rejects_unexpected_files_in_wasm_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = self._seed_dist(root)
            (dist / "wasm" / "unhashed.js").write_text("console.log(1)", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected files remain"):
                package_pages(
                    dist, root / "wasm-out", "https://assets.deepclone.lexo.cool"
                )

    def test_rejects_missing_security_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = self._seed_dist(root)
            (dist / "_headers").write_text("/*\n  X-Frame-Options: DENY\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Content-Security-Policy"):
                package_pages(dist, root / "wasm-out", "https://assets.deepclone.lexo.cool")

    def test_rejects_oversized_static_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = self._seed_dist(root)
            with (dist / "oversized.bin").open("wb") as handle:
                handle.seek(PAGES_MAX_FILE_BYTES)
                handle.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "25 MiB"):
                package_pages(dist, root / "wasm-out", "https://assets.deepclone.lexo.cool")

    def test_retains_active_and_previous_wasm_objects(self):
        objects = [
            {"key": "wasm/deepclone-1111.wasm", "last_modified": "2026-11-24T00:00:00Z"},
            {"key": "wasm/deepclone-3333.wasm", "last_modified": "2026-11-26T00:00:00Z"},
            {"key": "wasm/deepclone-2222.wasm", "last_modified": "2026-11-25T00:00:00Z"},
            {"key": "other/keep.txt", "last_modified": "2026-11-20T00:00:00Z"},
        ]
        self.assertEqual(
            stale_wasm_keys(objects, "wasm/deepclone-3333.wasm", keep=2),
            ["wasm/deepclone-1111.wasm"],
        )


if __name__ == "__main__":
    unittest.main()
