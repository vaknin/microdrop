#!/usr/bin/env python3

"""Download the `small.en` faster-whisper model into the local cache."""

from faster_whisper import WhisperModel


def main() -> None:
    print("Initializing faster-whisper model (small.en, int8 on cpu)...")
    WhisperModel(
        "small.en",
        device="cpu",
        compute_type="int8",
    )
    print("Model available in local cache.")


if __name__ == "__main__":
    main()
