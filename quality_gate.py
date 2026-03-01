#!/usr/bin/env python3
"""CLI entrypoint wrapper for model quality gate."""

from whisper_ja.cli.quality_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
