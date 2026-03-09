#!/usr/bin/env python3
"""CLI entrypoint wrapper for serving export + benchmark pipeline."""

from whisper_ja.cli.serving import main


if __name__ == "__main__":
    raise SystemExit(main())
