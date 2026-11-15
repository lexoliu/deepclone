"""Generate the speaker frontend's Slaney filter bank from pinned librosa."""

import argparse
import json
from pathlib import Path

import librosa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # These parameters are fixed by Qwen3-TTS extract_speaker_embedding.
    filters = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 256,
        "weights": librosa.filters.mel(
            sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000
        ).tolist(),
    }
    args.output.write_text(json.dumps(filters) + "\n")


if __name__ == "__main__":
    main()
