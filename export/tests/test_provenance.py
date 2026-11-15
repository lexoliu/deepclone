"""Reject incompatible resumptions before compiling or loading model weights."""

import tempfile
import unittest
from pathlib import Path

from provenance import snapshot, verify_or_record


class Provenance(unittest.TestCase):
    def test_source_content_and_shapes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            weights = source / "model.safetensors"
            weights.write_bytes(b"test weights")
            settings = {"context": 128, "chunk_layers": 4, "weight_bits": 16}
            original = snapshot(source, settings)
            verify_or_record(output, original, resume=False)
            verify_or_record(output, snapshot(source, settings), resume=True)
            with self.assertRaisesRegex(ValueError, "settings"):
                verify_or_record(
                    output, snapshot(source, {**settings, "context": 256}), resume=True
                )
            weights.write_bytes(b"new weights")
            with self.assertRaisesRegex(ValueError, "source"):
                verify_or_record(output, snapshot(source, settings), resume=True)
            with self.assertRaisesRegex(ValueError, "exporter"):
                verify_or_record(output, {**original, "exporter": {}}, resume=True)

    def test_untracked_output_cannot_be_adopted(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "without export-inputs"),
        ):
            verify_or_record(Path(directory), {}, resume=True)


if __name__ == "__main__":
    unittest.main()
