#!/usr/bin/env python3
"""CLI entrypoint wrapper for HF model inference."""

from whisper_ja.cli.infer_from_hf import main


if __name__ == "__main__":
    raise SystemExit(main())
