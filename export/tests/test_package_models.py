"""Public archives contain only selected inference files and no symlinks."""

import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from package_models import inputs, package


class Packaging(unittest.TestCase):
    def test_deterministic_regular_files_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformers, audio = root / "transformers", root / "audio"
            transformers.mkdir()
            audio.mkdir()
            manifest = {
                "weight_bits": 16,
                "text_embedding": "embedding.mlmodelc",
                "codec_frame": "embedding.mlmodelc",
                "codec_embeddings": [],
                "predictor_projection": None,
                "decoders": {},
            }
            (transformers / "transformers.json").write_text(json.dumps(manifest))
            for name in ["config.json", "tokenizer.json"]:
                (transformers / name).write_text("{}")
            shared = root / "shared"
            shared.mkdir()
            (shared / "weight.bin").write_bytes(b"fixture weights")
            (transformers / "embedding.mlmodelc").symlink_to(shared)
            (audio / "mel.json").write_text("{}")
            for component, precision in [
                ("encoder", "fp32"),
                ("speaker", "fp32"),
                ("decoder", "fp16"),
            ]:
                (audio / f"{component}.mlmodelc").symlink_to(shared)
                (audio / f"{component}.contract.json").write_text(
                    json.dumps(
                        {"compute_precision": precision, "execution": "stateful_frame"}
                    )
                )
            (audio / "reference.wav").write_bytes(b"private fixture")
            first = package(transformers, audio, root / "first.tar.zst")
            second = package(transformers, audio, root / "second.tar.zst")
            self.assertEqual(first["sha256"], second["sha256"])
            unpacked = subprocess.check_output(
                ["zstd", "-d", "-c", str(root / "first.tar.zst")]
            )
            with tarfile.open(fileobj=io.BytesIO(unpacked)) as archive:
                self.assertTrue(all(member.isfile() for member in archive))
                self.assertNotIn("audio/reference.wav", archive.getnames())
                self.assertEqual(
                    archive.extractfile("audio/encoder.mlmodelc/weight.bin").read(),
                    b"fixture weights",
                )
            manifest["weight_bits"] = 8
            manifest["decoders"] = {
                "talker": {"chunks": [], "heads": [], "sequence_lengths": [1, 16]}
            }
            (transformers / "transformers.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "batched waveform decoder"):
                inputs(transformers, audio)
            (audio / "decoder.contract.json").write_text(
                json.dumps(
                    {
                        "compute_precision": "fp16",
                        "execution": "stateful_frame",
                        "batch_frames": [1],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "batched waveform decoder"):
                inputs(transformers, audio)
            (audio / "decoder.contract.json").write_text(
                json.dumps(
                    {
                        "compute_precision": "fp16",
                        "execution": "stateful_frame",
                        "batch_frames": [1, 4, 12],
                    }
                )
            )
            self.assertIn(
                "transformers/embedding.mlmodelc/weight.bin",
                inputs(transformers, audio),
            )
            manifest["decoders"] = {}
            manifest["weight_bits"] = 4
            (transformers / "transformers.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "8-bit or 16-bit"):
                inputs(transformers, audio)
            manifest["weight_bits"] = 16
            manifest["text_embedding"] = "../private"
            (transformers / "transformers.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Invalid model path"):
                inputs(transformers, audio)


if __name__ == "__main__":
    unittest.main()
