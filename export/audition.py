"""Write a dual-mono listening copy without changing the generated waveform."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("A listening copy requires a mono source waveform")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps the original synthesis artifact intact.
    with args.output.open("xb") as output:
        sf.write(
            output, np.repeat(audio, 2, axis=1), rate, format="WAV", subtype="FLOAT"
        )


if __name__ == "__main__":
    main()
