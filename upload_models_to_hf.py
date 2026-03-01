#!/usr/bin/env python3
"""CLI entrypoint wrapper for uploading models to HF Hub."""

from whisper_ja.cli.upload_models_to_hf import main


if __name__ == "__main__":
    raise SystemExit(main())
