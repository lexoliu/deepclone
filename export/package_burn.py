"""Prepare pinned Qwen weights and complete architecture metadata for Burn."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import librosa
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from transformers import AutoTokenizer


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(size, source, output):
    pins = json.loads(Path(__file__).with_name("sources.json").read_text())[size]
    for item in pins["files"]:
        path = source / item["path"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError(f"Pinned source verification failed: {item['path']}")
    output.mkdir(parents=True, exist_ok=False)
    config = Qwen3TTSConfig.from_dict(json.loads((source / "config.json").read_text()))
    # Expanding configuration defaults makes the runtime independent of the
    # Python library's implicit speaker and codec architecture defaults.
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    shutil.copyfile(source / "speech_tokenizer/config.json", output / "codec.json")
    shutil.copyfile(source / "model.safetensors", output / "speech.safetensors")
    shutil.copyfile(
        source / "speech_tokenizer/model.safetensors", output / "codec.safetensors"
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.backend_tokenizer.save(str(output / "tokenizer.json"))
    # Parameters are fixed by Qwen3-TTS extract_speaker_embedding.
    mel = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 256,
        "weights": librosa.filters.mel(
            sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000
        ).tolist(),
    }
    (output / "mel.json").write_text(json.dumps(mel) + "\n")
    files = [
        {"path": p.name, "sha256": digest(p), "size": p.stat().st_size}
        for p in sorted(output.iterdir())
    ]
    manifest = {
        "format": "deepclone-burn-v1",
        "source": {key: pins[key] for key in ("repository", "revision", "license")},
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=["0.6b", "1.7b"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.size, args.source, args.output)


if __name__ == "__main__":
    main()
