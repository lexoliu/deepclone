"""Preserve real codec prefixes and independent native reproduction coverage."""

import subprocess
import sys
import unittest
from pathlib import Path

from probe_interleaving import inputs


class Interleaving(unittest.TestCase):
    def test_prefixes_cross_blocks_without_padding_or_lost_frames(self):
        groups, frames, reference = 3, 655, 151
        values = list(range(groups * frames))
        calls = list(
            inputs(
                {"groups": groups, "values": values, "reference_frames": reference},
                325,
                12,
            )
        )
        cursor = reference
        decoded = []
        for call in calls:
            self.assertEqual(call["context"], max(0, cursor // 300 * 300 - 25))
            self.assertLessEqual(call["frames"], 325)
            self.assertEqual(len(call["values"]), groups * call["frames"])
            for frame in range(cursor - call["context"], call["frames"]):
                decoded.extend(
                    call["values"][group * call["frames"] + frame]
                    for group in range(groups)
                )
            cursor = call["context"] + call["frames"]
        self.assertEqual(cursor, frames)
        self.assertEqual(decoded, values[reference * groups :])
        with self.assertRaisesRegex(ValueError, "capacity"):
            list(
                inputs(
                    {"groups": groups, "values": values, "reference_frames": reference},
                    324,
                    12,
                )
            )

    @unittest.skipUnless(
        sys.platform == "darwin", "Core ML native probe requires macOS"
    )
    def test_independent_swift_reproduction_compiles(self):
        subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-typecheck",
                "-parse-as-library",
                str(Path(__file__).with_name("coreml_interleaving.swift")),
            ],
            check=True,
            timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
